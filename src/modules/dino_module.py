import torch
import torchvision
import lightning as pl
import torch.nn.functional as F



import torch.nn as nn

import copy
from torchvision.utils import make_grid

# dataset
from dataset import ImageDataset, TrainTransform, ValTransform
from torch.utils.data import DataLoader

# model
from pytorch_lightning import LightningModule

from models import MultiCropWrapper, DINOHead
from ..tools.losses import DINOLoss
from ..tools.training_utils import  trunc_normal_


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        """
        x: list of input image tensors
        """
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0) # [2, 10] for student, [2] for teacher
        start_idx = 0
        out = []
        for end_idx in idx_crops:
            out += [self.backbone(torch.cat(x[start_idx:end_idx]))]
            start_idx = end_idx
        return self.head(torch.cat(out))

class DINOModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        model = vits_dict[hparams.arch] # TODO
        
        student_backbone = model(patch_size=hparams.patch_size,
                                 drop_path_rate=hparams.drop_path_rate)
        self.teacher_backbone = model(patch_size=hparams.patch_size)

        student_head = DINOHead(student_backbone.embed_dim, hparams.out_dim,
                                hparams.norm_last_layer)
        teacher_head = DINOHead(self.teacher_backbone.embed_dim, hparams.out_dim)

        self.student = MultiCropWrapper(student_backbone, student_head)
        if hparams.pretrained_path: # fine-tune from pretrained dino
            print(f'loading pretrained model from {hparams.pretrained_path} ...')
            ckpt = torch.load(hparams.pretrained_path, map_location='cpu')
            self.student.load_state_dict(ckpt['teacher'])
        self.teacher = MultiCropWrapper(self.teacher_backbone, teacher_head)
        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())

        # teacher is not trained
        for p in self.teacher.parameters(): p.requires_grad = False

        self.loss = DINOLoss(hparams.out_dim,
                             hparams.local_crops_number+2,
                             hparams.warmup_teacher_temp,
                             hparams.final_teacher_temp,
                             hparams.warmup_teacher_temp_epochs,
                             hparams.num_epochs)

    def setup(self, stage=None):
        print('loading image paths ...')
        self.train_dataset = ImageDataset(hparams.root_dir, 'train')
        print(f'{len(self.train_dataset.image_paths)} image paths loaded!')

        self.val_dataset = copy.deepcopy(self.train_dataset)
        self.val_dataset.split = 'val'

        self.train_dataset.transform = \
            TrainTransform(hparams.global_crops_scale,
                           hparams.local_crops_scale,
                           hparams.local_crops_number)
        self.val_dataset.transform = ValTransform()

    def configure_optimizers(self):
        regularized, not_regularized = [], []
        for n, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if n.endswith(".bias") or len(p.shape) == 1:
                not_regularized.append(p)
            else:
                regularized.append(p)
        param_groups = [{'params': regularized},
                        {'params': not_regularized, 'weight_decay': 0.}]

        self.lr = hparams.lr * (hparams.batch_size*hparams.num_gpus/256)
        opt = torch.optim.AdamW(param_groups, self.lr)

        return opt

    def train_dataloader(self):
        self.loader = DataLoader(self.train_dataset,
                                 shuffle=True,
                                 num_workers=hparams.num_workers,
                                 batch_size=hparams.batch_size,
                                 pin_memory=True,
                                 drop_last=True)

        # define schedulers based on number of iterations
        niter_per_ep = len(self.loader)
        self.lr_sch = cosine_scheduler(self.lr, 1e-6, hparams.num_epochs, niter_per_ep//hparams.num_gpus,
                                       hparams.warmup_epochs)
        # weight decay scheduler
        self.wd_sch = cosine_scheduler(hparams.weight_decay_init, hparams.weight_decay_end,
                                       hparams.num_epochs, niter_per_ep//hparams.num_gpus)
        # momentum scheduler
        self.mm_sch = cosine_scheduler(hparams.momentum_teacher, 1.0,
                                       hparams.num_epochs, niter_per_ep//hparams.num_gpus)

        return self.loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=hparams.num_workers,
                          batch_size=1, # validate one image
                          pin_memory=True)

    def training_step(self, batch, batch_idx):
        """
        batch: a list of "2+local_crops_number" tensors
               each tensor is of shape (B, 3, h, w)
        """
        opt = self.optimizers()
        # update learning rate, weight decay
        for i, param_group in enumerate(opt.param_groups):
            param_group['lr'] = self.lr_sch[self.global_step]
            if i == 0: # only the first group is regularized
                param_group['weight_decay'] = self.wd_sch[self.global_step]

        teacher_output = self.teacher(batch[:2])
        student_output = self.student(batch)
        loss = self.loss(student_output, teacher_output, self.current_epoch)

        opt.zero_grad()
        self.manual_backward(loss)
        # clip gradient
        if hparams.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), hparams.clip_grad)
        # cancel gradient for the first epochs
        if self.current_epoch < hparams.ep_freeze_last_layer:
            for n, p in self.student.named_parameters():
                if "last_layer" in n:
                    p.grad = None
        opt.step()

        # EMA update for the teacher
        m = self.mm_sch[self.global_step]
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_((1-m)*ps.data)

        self.log('rates/lr', opt.param_groups[0]['lr'])
        self.log('rates/weight_decay', opt.param_groups[0]['weight_decay'])
        self.log('rates/momentum', m)
        self.log('train/loss', loss, True)

    def validation_step(self, batch, batch_idx):
        img_orig, img_norm = batch

        w_featmap = img_norm.shape[-1] // hparams.patch_size
        h_featmap = img_norm.shape[-2] // hparams.patch_size

        atts = self.teacher_backbone.get_last_selfattention(img_norm)
        atts = atts[:, :, 0, 1:].reshape(1, -1, h_featmap, w_featmap)
        atts = torch.nn.functional.interpolate(atts,
                    scale_factor=hparams.patch_size, mode="nearest")[0] # (6, h, w)

        return {'attentions': atts, 'img': img_orig}

    def validation_epoch_end(self, outputs):
        atts = outputs[0]['attentions']

        tb = self.logger.experiment
        tb.add_image('image', outputs[0]['img'][0], self.global_step)
        atts_vis = [att2img(att) for att in atts]
        tb.add_image('attentions', make_grid(atts_vis, nrow=3), self.global_step)


