import torch
from torch.utils.data import Dataset
import pandas as pd
import anndata
import numpy as np
import pickle
import os
import time
import logging
from ..tools.constants import TEST_EXPERIMENTS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    #level=logging.DEBUG,
    )
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    def __init__(self, anndata_file, parquet_file, obsm_key, emb_align="pheno_embedding", mode='train'):
        logger.info("Initializing the MultimodalDataset...")
        self.emb_align = emb_align
        self.obsm_key = obsm_key
        
        # Load anndata file
        logger.info("Loading Tx file...")
        self.tx_data = anndata.read_h5ad(anndata_file)

        self.tx_data = self.tx_data[self.tx_data.obs['has_control'] == False]

        # if train, filter out all TEST_EXPERIMENTS from obs of tx_data in column experiment_id
        if mode == 'train':
            self.tx_data = self.tx_data[~self.tx_data.obs['experiment_id'].isin(TEST_EXPERIMENTS)]
        else:
            self.tx_data = self.tx_data[self.tx_data.obs['experiment_id'].isin(TEST_EXPERIMENTS)]

        
        # Load parquet file
        logger.info("Loading Ph file...")
        self.ph_data = pd.read_parquet(parquet_file)

        self.create_ph_mapping()
        

        # Match indices using vectorized operations
        self.match_indices()


    def create_ph_mapping(self):
        logger.info("Creating phenotype mapping...")
        # Convert embeddings list into a numpy array for vectorized operations
        self.ph_data[self.emb_align] = self.ph_data[self.emb_align].apply(np.array)

        # Create dictionary directly from DataFrame
        self.ph_mapping = {}
        for index, row in self.ph_data.iterrows():
            # Ensure keys are not numpy arrays. Convert them to strings or other primitives if necessary.
            smiles = row['smiles'] if isinstance(row['smiles'], str) else str(row['smiles'])
            #print(row['concentration'])
            concentration = float(row['original_concentration']) # if isinstance(row['concentration'], (int, float, str)) else str(row['concentration'])
            # check if it exists in dict
            if (smiles, concentration) in self.ph_mapping.keys():
                self.ph_mapping[(smiles, concentration)].append(torch.tensor(row[self.emb_align], dtype=torch.float32))
            else :
                self.ph_mapping[(smiles, concentration)] = [torch.tensor(row[self.emb_align], dtype=torch.float32)]
        logger.info(f"Phenotype mapping created with {len(self.ph_mapping)} entries.")
            

    def match_indices(self):
        
        # Create a set of (canonical_smiles, concentration) pairs from filtered_adata
        adata_pairs = set(zip(self.tx_data.obs['canonical_smiles'], self.tx_data.obs['treatment_concentration']))

        # Create a set of (smiles, concentration) pairs from df
        df_pairs = set(zip(self.ph_data['smiles'], self.ph_data['original_concentration']))
        # Identify the intersection of these sets to find shared (smiles, concentration) pairs
        shared_pairs = adata_pairs.intersection(df_pairs)

        # Create a DataFrame from the shared pairs for easier filtering
        shared_pairs_df = pd.DataFrame(list(shared_pairs), columns=['canonical_smiles', 'treatment_concentration'])

        # Merge the shared pairs with filtered_adata to filter rows
        self.tx_data = self.tx_data[
            self.tx_data.obs.set_index(['canonical_smiles', 'treatment_concentration']).index.isin(shared_pairs_df.set_index(['canonical_smiles', 'treatment_concentration']).index)
        ].copy()
        # reset index
        self.tx_data.obs.reset_index(inplace=True)
        self.indices = self.tx_data.obs.index.tolist()


        # repeat indices 3 times
        self.indices = self.indices * 3
        np.random.shuffle(self.indices)

        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        
        tx_idx = self.indices[idx]
        tx_embedding = torch.tensor(self.tx_data.obsm[self.obsm_key][tx_idx], dtype=torch.float32)
        
        tx_smiles = self.tx_data.obs['canonical_smiles'].iloc[tx_idx]
        tx_concentration = self.tx_data.obs['treatment_concentration'].iloc[tx_idx]
        
        ph_embedding = self.ph_mapping[(tx_smiles, float(tx_concentration))]
        
        ph_embedding = ph_embedding[np.random.randint(0, len(ph_embedding))]
        return [tx_embedding, ph_embedding]

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        np.random.shuffle(self.indices)

def multimodal_collate_fn(batch):
    views1, views2 = zip(*[(item[0], item[1]) for item in batch])
    views1 = torch.stack(views1)
    views2 = torch.stack(views2)
    return views1, views2