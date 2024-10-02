import torch
import torchvision
import lightning as pl
import torch.nn.functional as F


class SimCLR(pl.LightningModule):
    def __init__(self, model, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        if model is not None:
            self.model = model
        else:
            ## This part is for testing / debugging only
            hidden_dim = 256
            self.model = torchvision.models.resnet18(
                pretrained=False, num_classes=4 * hidden_dim
            )  # num_classes is the output size of the last linear layer
            # The MLP for g(.) consists of Linear->ReLU->Linear
            self.model.fc = torch.nn.Sequential(
                self.model.fc,  # Linear(ResNet output, 4*hidden_dim)
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(4 * hidden_dim, hidden_dim),
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        mols, imgs = batch
        #imgs = torch.cat(imgs, dim=0)

        # Encode all images
        mol_feats, img_feats = self.model(mols, imgs)
        # Calculate cosine similarity
        #cos_sim = torch.nn.functional.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        cos_sim = torch.nn.functional.cosine_similarity(mol_feats, img_feats, dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")