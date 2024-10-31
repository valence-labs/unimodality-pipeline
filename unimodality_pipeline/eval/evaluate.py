import os
import torch
import anndata

from unimodality_pipeline.datasets.basic_dataset_module import MultiModalDataModule
from unimodality_pipeline.eval.post_process import center_scale_process
from unimodality_pipeline.eval.bmdb_eval import compute_bmdb_scores



def evaluate(args, system, trainer, logger, wb):
    # Parsing data files in the prediction directory
    files = []
    for path in os.listdir(args.tx_pred_data_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(args.tx_pred_data_dir, path)):
            files.append(path)
    
    for f in files:
        # Prepare data
        tx_pred_data_path = os.path.join(args.tx_pred_data_dir, f)
        data_module = MultiModalDataModule(
            multimodal_tx_data_path = None, 
            multimodal_tx_obsm_key = None, 
            multimodal_ph_data_path = None, 
            evaluation_tx_data = tx_pred_data_path, 
            evaluation_tx_obsm_key = args.pred_obsm_key, 
            batch_size = args.batch_size, 
            num_workers = args.num_workers,
            evaluation_tx_n_samples = args.n_eval_samples,
        )
        predictions = trainer.predict(system, datamodule=data_module)
        logger.info(f">> Converting predictions ...")
        # Concatenate predictions
        all_predictions = torch.cat(predictions, dim=0)

        # Convert to NumPy array if needed
        all_predictions_numpy = all_predictions.cpu().numpy()
        
        adata = anndata.read_h5ad(tx_pred_data_path)
        
        final_obsm_key = args.eval_obsm_key.split('_')[-1] + args.save_emb_name
        adata.obsm[final_obsm_key] = all_predictions_numpy

        if args.write_embs == True :
            logger.info(f">> Writing the embeddings to files...")
            adata.write_h5ad(args.tx_pred_data_path, as_dense='X')
        
        centered_embs = center_scale_process(adata, final_obsm_key, 'dataset_batch_num')
        metadata = adata.obs
        
        logger.info(f">> Computing Bmdb scores...")
        datasets, results = compute_bmdb_scores(metadata, centered_embs)
        good_results = []

        logger.info(f">> Logging scores to wandb...")
        for dataset, result in zip(datasets, results):
            wb.log({'l1000_' + dataset: result})
            if 'CORUM' in dataset :
                good_results.append(result)
            if 'HuMAP' in dataset :
                good_results.append(result)
            if 'Reactome' in dataset :
                good_results.append(result)
            if 'SIGNOR' in dataset :
                good_results.append(result)
            if 'StringDB' in dataset :
                good_results.append(result)
        wb.log({f"{'.'.join(w for w in f.split('.')[:-1])}_avg":sum(good_results)/len(good_results)})