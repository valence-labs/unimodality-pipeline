# Unimodality pipeline
This repository contains the training/prediction code for the unimodality project. It is intended to serve as a backbone for the project's experiments/evaluation tasks. It is primarily based on [Pytorch-Lightning](link-here) framework. 

## Setup
First, make sure to clone the **exact conda environment** used to reproduce these experiments (you can do it b y running the command `conda create -f env.yml -n <CUSTOM-NAME>`). Then, a quick way to start contributing to the project would be to fork the repository and proceed with **local editable installation** by running the following command at the base folder of the project:
```bash
pip install -e .
```
Once installed, files can be modified via any code editor and changes will be immediately reflected on the program's execution.

## Code organization
The code has been designed to be as much modular as possible, hence facilitating its maintenance/continuous improvement. There are three 3 main folders:
- notebooks: contains code for generating test anndata. Not very important for the project.
- scripts: contains sh scripts showing how to launch experiments in slurm environments. Each script corresponds to a specific experiment.
- unimodality_pipeline: folder containing the main source code.
The source code in the `unimodality_pipeline` folder has been subdivided to 5 python modules:
- datasets: contains pytorch-lightning data modules (file `basic_dataset_module.py`), as well as some dataset-specific utilities (ile `basic_dataset.py`).
- models: contains source code for neural networks used to build unimodality models.
- setups: contains pytorch-lightning training modules (like `clip_module.py`). These files define the entire plan for training models.
- tools: contains utilities used by other modules. As for example, Clip-loss loss functions are defined within `clip_losses.py` file and are used by the training module `clip_module.py`.
- tests: contains the python scripts used for training (`run_clip.py`) and inference (`run_inference.py`). These files are called by the `.sh` scripts located in the outer folder `scripts`.
So far, a simple clip-based model has been implemented to serve as example. To add a new experiment involving new system/loss/dataset, one probably has to follow the steps below:
1- Define a Dataset class and a dataset module in the `datasets` folder (or use the existing one(s)). Update the corresponding `__init__.py` in case of adding new classes/variables.
2- Define a neural architecture in the `models` folder (or use the existing one(s)). Update the corresponding `__init__.py` in case of adding new classes/variables.
3- Define a training module in the `setups` folder. Update the corresponding `__init__.py` iwith the name of the newly added module.
4- Potentially add new functions in the `tools` folder (new loss functions/routines used by other modules). Update the corresponding `__init__.py` in case the newly added classes/variables need to be exposed.
5- Add a runner script in the `tests` folder. The corresponding `__init__.py` does not need to be changed as the newly added functions are not exposed to other modules. <br />
However, in case of adding a new python module (folder) to the project `unimodality_pipeline`, the `__init__.py` needs to be changed to import the newly added module (in case the runner needs to import it of course).

## Reproducing the current experiments
### Scripts
Existing scripts can be root from the base directory of the project, either on interactive mode or differed-job mode. 6 scripts are ready to use:
- run_clip.sh: runs clip training script with/without inference.
- run_inference.sh: runs clip inference script.
- run_clip_frozen_ph_encoder.sh: runs clip training script with frozen phenomics encoder (with/without inference).
- run_clip_frozen_tx_encoder.sh: runs clip training script with frozen transcriptomics encoder (with/without inference).
- run_clip_disabled_ph_encoder.sh: runs clip training script without phenomics encoder (with/without inference).
- run_clip_disabled_tx_encoder.sh: runs clip training script without transcriptomics encoder (with/without inference).
To run the regular clip training with inference on biohive, one has to run the command:
```bash
sbatch scripts/run_clip.sh
```
### Data
Two datasets are used for training. They can be found at the directory `/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality`:
1 -  Transcriptomics: file `huvec_compounds.h5ad`.
2 -  Phenomics: file `HUVEC-tvn_v11_prox_bias_reduced_PHENOM1-2023-09-28_smiles_4splits_v3_filtered_transcriptomics_molphenix_embeds.parquet`.

Regarding inference, replogle dataset has been modified by adding a random-float-valued-column `tests` that emulates embeddings. It is located at: `/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/data/replogle_2022.h5ad`. In case this file is not accessible, it can be re-created using the jupyter notebook `notebooks/prepare_replogle_data.ipynb`.

## Quick dive in some python modules
### Datasets
This module implements the necesary structures needed for data loading. The file `basic_dataset.py` contains regular `Dataset` classes. The file `basic_dataset_module.py` implements the data module inteded for the training module `ClipModule`. Below is a summary of its main methods:
- __init__: constructor.
- setup: called each time the trainer enters un new stage ('fit', 'validate', 'predict'). Caller sets the parameter `stage` to the corresponding value. We use this method to instantiate datasets 'on demand' as we don't necessarily need them in all scenarios.
- train_dataloader: returns the train set dataloader.
- val_dataloader: returns the validation set dataloader.
- predict_dataloader: returns the prediction set dataloader.
- teardown: called every time a stage ends. Caller sets the parameter `stage` to the corresponding value. This function is used for freeing memory.
There are of courses other methods defined in the `LightningDataModule` class that are not overriden here. More information is available [here](link).

**Important note:** We used `setup` callbacks to load datasets because we don't need them all for training and inference. `prepare_dataset` callback can be used too, but unlike the former, it is called only once. `setup` is called at different stages of training, allowing for loading data only when it is needed.
### Setups
The `clip_module.py` implements the unimodality models' training pipeline using the regular clip loss function. This file should be used as example to implement other training pipelines (such as `dino_module.py` and `simclr_module.py`). Below is the description of the main methods:
- __init__: constructor of the class. Primarily used for saving the hyper parameters passed as argument (very important) as well as for building the models/other necessary structures. 
- set_encoder_mode: used to set one of the encoders (phenomics, transcriptomics) to a frozen/trainable state.
- setup: called everytime the trainer enters a new stage. Caller sets the parameter `stage` to the corresponding value.
- teardown: called everytime the trainer leaves a stage. Caller sets the parameter `stage` to the corresponding value.
- forward: though this function is not necessary for training, it is called when `prediction_step` is called.
- training_step: defines how the training loss is computed. Also logs it.
- validation_step: defines how the validation loss and/or other metrics are computed. Also logs the results.
- predition_step: calls the `forward` method.
- configure_optimizers: implements the optimizers.

Not all of the `LightningModule` class' methods have been overriden by the `ClipModule` class. More information is available [here](link).
**Important note:** Classes derived from `LightningModule` can have any number of parameters. However, passing only the hyperparameters as an  namespace, followed by a call to `save_hyperparameters` function ensures the ability of loading the module from a checkpoint without knowing its class attributes (see file `run_inference.py`, where the ClipModule class is loaded from disk). Failing to do so (such as passing complex structures which cannot be saved as parameters) prevents the loading of the module without calling its `__init__` method, hence necessitating the instantiation of the constructor's arguments. For this reason, we recommend to only pass the hyperparameters to the module and move the models' building logic inside it. 
### Tools
- constants.py: this files contains global variables that all likely to be used by other files/modules.
- clip_losses.py: this files implements clip loss, which is used by the `ClipModule` class.
**Important note:** As mentioned above, some training settings involve freezing/removing one of the tx/ph encoders (following the team's request). Encoder removal enforces the use of the embedding as is in the loss function, hence causing a problem of dimensions when it comes to matrix multiplication. To overcome this problem, additional multiplications were performed to get rid of the embeddings' size, hence altering the original clip loss' logic. One possible solution would be to add trainable/frozen projection heads to align the dimensions of loss function's arguments (see example of projection head implementation [here](link)).

### Tests
This folder contains python runners that are used for training and inference. They can be used as examples to implement additonal scripts:
- run_clip.py: Used mainly for training with/without inference. Many training options are available (see the file). The script uses pytorch-lightning's trainer to train and save models with their hyper-parameters. It currently supports single and multiple gpus.
- run_inference.py: Used mainly for inference. Many training options are available, namely the one controlling the number of samples to use (`--n_samples` argument). The script uses pytorch-lightning's trainer to load a previously saved model from the disk and use it for predictions. It currently supports single and multiple gpus.

## Next steps
The following sections describe some ideas regarding necessary/possible improvements that need to be brought to the current code.
### datasets
Currently, there is a single complex data module `basic_dataset_module.py` that uses both classes `MultiModalDataset` and `TxDataset`. This data module is also used by inference script, in which case it does not load the training sets (works also for prediction set, which is not loaded if we do not want to predict during training). To do so, data reading logic was placed in `setup` callback, hence making it possible to load data only when you need it ('fit', 'validate' or 'predict' stages). As small improvement, we could split the logic and have multiple data modules instead. The main benefit would be to reduce the complexity of the datamodule classes, and use the `prepare_data` callback intead of `setup`. The main drawback is that we don't know (yet) whether pytorch-lightning's `Trainer` class supports multiple modules or not. Let's find out!.
### models
There is room for improvement as we currently use simple MLPs as encoders. More complex architectures (GNNs, CNNs, pretrained models...) could be added as well. 
### setups
So far, only clip-loss training setup has been implemented. More baselines (SimClr, Dino, ...) are likely to be needed in a near future. Some features, such as optimizers and strategy, are currently hard-coded and should be dynamically instantiated instead (using passed arguments). We should be able to support different optimizers/hardware accelerators (such as Deepspeed). Once done, runners (`tests` folder) should be modified accordingly. 
### tests
Current runners use a specific lightning module (`ClipModule`) for both training and inference. We could add more runners as more training modules are implemented, or modify the current code to make a runner module-agnostic (using a dictionary for instance, we could have a mapping string <-> LightningModule class and dynamically instantiate a module using a string passed as argument...). 
### tools
This python module is intended for tools/utilities, namely loss functions and various routines. So far, only clip loss is implemented, but more losses can be added (the Hopfield variant, for instance). If many losses are to be used, a possible solution would be to group them by family (Clip loss family for instance), to avoid having exponentially-growing python files. At last but not least, **the existing ClipLoss class needs to be revisited**, as mentioned in previous sections.
## References

```
@misc{https://doi.org/10.48550/arxiv.2110.08460,
  doi = {10.48550/ARXIV.2110.08460},
  url = {https://arxiv.org/abs/2110.08460},
  author = {Li, Tianda and El Mesbahi, Yassir},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Short Study on Compressing Decoder-Based Language Models},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

- Loading lightning module from checkpoint: https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#lightningmodule-from-checkpoint 
- Implementing Clip with lightning: https://wandb.ai/manan-goel/coco-clip/reports/Implementing-CLIP-With-PyTorch-Lightning--VmlldzoyMzg4Njk1
- Using lightning for inference: https://pytorch-lightning.readthedocs.io/en/1.6.5/common/production_inference.html
-
