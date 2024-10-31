from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


from .basic_dataset import (
    TxDataset,
    MultimodalDataset,
    multimodal_collate_fn
)
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        )
logger = logging.getLogger(__name__)

class MultiModalDataModule(LightningDataModule):
    def __init__(
        self, 
        multimodal_tx_data_path, 
        multimodal_tx_obsm_key, 
        multimodal_ph_data_path, 
        evaluation_tx_data: str, 
        evaluation_tx_obsm_key: str, 
        evaluation_tx_filter_controls:bool = False, 
        evaluation_tx_n_samples:int = 1000,
        batch_size: int = 128,
        num_workers: int = 2,
        ):
        super().__init__()
        self.multimodal_tx_data = multimodal_tx_data_path
        self.multimodal_tx_obsm_key = multimodal_tx_obsm_key
        self.multimodal_ph_data = multimodal_ph_data_path
        self.evaluation_tx_data = evaluation_tx_data
        self.evaluation_tx_obsm_key = evaluation_tx_obsm_key
        self.evaluation_tx_filter_controls = evaluation_tx_filter_controls
        self.evaluation_tx_n_samples = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None

    def setup(self, stage: str):
        if stage == 'fit':
            logger.info(f">> Loading training set")
            self.train_dataset = MultimodalDataset(self.multimodal_tx_data, self.multimodal_ph_data, self.multimodal_tx_obsm_key, mode='train')
            logger.info(f">> Loading validation set")
            self.val_dataset = MultimodalDataset(self.multimodal_tx_data, self.multimodal_ph_data,  self.multimodal_tx_obsm_key, mode='test')
        elif stage == 'predict':
            logger.info(f">> Loading prediction set")
            self.pred_dataset = TxDataset(
                self.evaluation_tx_data, 
                obsm_key=self.evaluation_tx_obsm_key, 
                filter_controls = self.evaluation_tx_filter_controls, 
                n_samples=self.evaluation_tx_n_samples
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=multimodal_collate_fn,
                          shuffle=True,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          collate_fn=multimodal_collate_fn,
                          shuffle=False,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          drop_last=True)


    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, 
                          shuffle=False,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          drop_last=False)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        if stage == 'fit':
            del self.train_dataset
            self.train_dataset = None
            del self.val_dataset
            self.val_dataset = None          
        elif stage == 'predict':
            del self.pred_dataset
            self.pred_dataset = None
