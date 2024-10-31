import torch
from torch.utils.data import Dataset
import pandas as pd
import anndata
import numpy as np
import logging
from ..tools.constants import TEST_EXPERIMENTS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    #level=logging.DEBUG,
    )
logger = logging.getLogger(__name__)



class TxDataset(Dataset):
    def __init__(self, anndata_file, obsm_key, filter_controls = False, n_samples=None):
        self.obsm_key = obsm_key
        
        # Load anndata file
        self.tx_data = anndata.read_h5ad(anndata_file)

        if filter_controls == True:
            self.tx_data = self.tx_data[self.tx_data.obs['has_control'] == False]
        if (n_samples is not None)  and (len(self.tx_data.obs) >  n_samples):
            self.tx_data = self.tx_data[:n_samples,:]
            logger.info(f"Keeping {len(self.tx_data)} samples...")
        # Reset index to ensure it's sequential
        self.tx_data.obs.reset_index(drop=True, inplace=True)

        # Use sequential indices
        self.indices = np.arange(len(self.tx_data)).tolist()
    def __len__(self):
        return len(self.tx_data)

    def __getitem__(self, idx):
        #tx_idx = self.indices[idx]
        return torch.tensor(self.tx_data.obsm[self.obsm_key][idx], dtype=torch.float32)




# pheno_embedding
# mp_image_embedding
# mp_dosage_embedding
class MultimodalDataset(Dataset):
    def __init__(self, anndata_file, parquet_file, obsm_key, emb_align="pheno_embedding", mode='train'):
        self.emb_align = emb_align
        self.obsm_key = obsm_key

        self.augment = False
        
        # Load anndata file
        self.tx_data = anndata.read_h5ad(anndata_file)

        self.tx_data = self.tx_data[self.tx_data.obs['has_control'] == False]

        # if train, filter out all TEST_EXPERIMENTS from obs of tx_data in column experiment_id
        if mode == 'train':
            self.tx_data = self.tx_data[~self.tx_data.obs['experiment_id'].isin(TEST_EXPERIMENTS)]
        else:
            self.tx_data = self.tx_data[self.tx_data.obs['experiment_id'].isin(TEST_EXPERIMENTS)]

        
        # Load parquet file
        self.ph_data = pd.read_parquet(parquet_file)

        self.create_ph_mapping()
        

        # Match indices using vectorized operations
        self.match_indices()


    def create_ph_mapping(self):
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
        if self.augment :
            obsm_key =  self.obsm_key + random.choice(['','CenterScale'])
        else :
            obsm_key = self.obsm_key
        tx_embedding = torch.tensor(self.tx_data.obsm[obsm_key][tx_idx], dtype=torch.float32)
        
        tx_smiles = self.tx_data.obs['canonical_smiles'].iloc[tx_idx]
        tx_concentration = self.tx_data.obs['treatment_concentration'].iloc[tx_idx]

        label = self.tx_data.obs['labels'].iloc[tx_idx]
        
        # Ensure label is a scalar
        if isinstance(label, (np.ndarray, list)):
            label = label.item()  # Convert to scalar if necessary

        label = torch.tensor(label, dtype=torch.long)
        
        ph_embedding = self.ph_mapping[(tx_smiles, float(tx_concentration))]
        
        ph_embedding = ph_embedding[np.random.randint(0, len(ph_embedding))]
        return [tx_embedding, ph_embedding, label]

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch
        np.random.shuffle(self.indices)

def multimodal_collate_fn(batch):
    views1, views2, labels = zip(*[(item[0], item[1], item[2]) for item in batch])
    views1 = torch.stack(views1)
    views2 = torch.stack(views2)
    labels = torch.stack(labels)
    return views1, views2, labels
