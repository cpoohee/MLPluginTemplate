# Introduction
This repo is an AI project template intended for AI generative models for music production.

Originally it was intended to be a short project aimed for submission for [Neural Audio Plugin Competition 2023](https://www.theaudioprogrammer.com/neural-audio).

There are 2 separate git repository for this plugin.
The first is for the machine learning code. The second is for the plugin code for running the trained model.

- https://github.com/cpoohee/MLPluginTemplate (this repo, the ML code base)
- https://github.com/cpoohee/NeuralPluginTemplate (Plugin code)

# Replication Instructions

## 1) Pre-requisites

- The ML code is created to run with Nvidia GPU (cuda) on Ubuntu 20.04 or Apple Silicon hardware (mps) in mind.
  - For installing cuda 11.8 drivers, see (https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
  - For Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1` as your environment variable when running python scripts, as not all pytorch ops are mps compatible. 
- Prepare at least 150GB of free hdd space. 
- The current dataset and cached files size used are about 50 GB. (nus-48e, vocalset, vctk) 

- Install Miniconda. [See guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
- For Apple Silicon, you will need to use Miniconda for mps acceleration. Anaconda is not recommended.

- [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### Install system libraries
For Ubuntu
```bash 
sudo apt-get update
sudo apt-get install libsndfile1
```

For Macos
  - install brew. See https://brew.sh/
```bash 
brew install libsndfile
```

## 2) Major Tech Stack Used for Development
- Pytorch with pytorch lightning for machine learning
- Hydra, for configurations
- MLFlow, for experiment tracking
- ONNX, for porting model to C++ runtime

## 3) Clone the repository

Go to your terminal, create a folder that you would like to clone the repo. 

Run the following command in your terminal to clone this repo. 
```bash 
git clone https://github.com/cpoohee/MLPluginTemplate
```

## 4) Create conda environment
Assuming, miniconda is properly installed, run the following to create the environment
```bash 
conda env create -f environment.yml
```

Activate the environment
```bash 
conda activate wave
```

## 5) Download Dataset (with examples)
Edit the scripts in `src/download_data.py` and target the code to download data to `/data/raw` folder.
The script will download the raw datasets and saves it onto `/data/raw` folder.
The script should also transcode the audio to `wav` format and select useful audio files to the `/data/interim` folder.

The sample download script is created for downloading the following datasets
- NUS-48E
- VocalSet
- VCTK


To run the download script, go to the project root folder

```bash 
cd MLPluginTemplate
```

Then, run the download script
```bash 
python src/download_data.py
``` 

* in case of failure especially from gdrive links, you try to delete contents of `/data/raw` and try again later. 
the scripts will skip the download if it detects the folder exists.


## 6) Download Pre-trained Models
Edit the download Pre-trained Models script for your needs.

Run the following, and it will download pre-trained models into the `/models/pre-trained` folder
```bash 
python src/download_pre-trained_models.py 
```


## 7) Pre-process Dataset
**Important!!**  Current pre-process code assumes an audio input and a labelled target format.
If you have a different input / target format, do modify the `src/process_data.py`. 
This is especially true for audio input and audio output targets.


The command below will pre-process NUS-48E, VCTK and vocalset datasets.
```bash 
python src/process_data.py process_data/dataset=nus_vocalset_vctk
```

Pre-processing will split the dataset into train/validation/test/prediction splits as stored into the `./data/processed` folder.
It also slices the audio into 5 sec long clips.

### (Optional) replace it with options:

* `process_data/dataset=nus` for pre-processing just the NUS-48E dataset
* `process_data/dataset=vctk` for pre-processing just the VCTK dataset

* See the `conf/process_data/process_root.yaml` for more detailed configurations.

## 8) Cache speech encodings (specific to AE models in this example)
Run the following 

```bash 
python src/cache_dataset.py model=autoencoder_speaker dataset=nus_vocalset_vctk
```

It will cache the downloaded pre-trained speaker encoder's embeddings.

### (Optional)
To use cuda (Nvidia)
```bash 
python src/cache_dataset.py model=autoencoder_speaker dataset=nus_vocalset_vctk process_data.accelerator=cuda
```

or mps (Apple silicon)
```bash 
python src/cache_dataset.py model=autoencoder_speaker dataset=nus_vocalset_vctk process_data.accelerator=mps
```

## 9) Dataloader
Modify the `src/datamodule/audio_dataloader.py` for your model input/target needs. 

Augmentation codes can be found here. 
Augmentation parameters should be added/modified in `conf/augmentations` yaml files.


## 10) Model
Add or modify pytorch lightning model codes under `src/model`.  
Add or modify model parameters `conf/model` yaml files.

Make sure the model input/target format from your dataloader matches your model requirements.

For example, this dataloader has audio x and y, followed by vectors and label.
```
def _shared_eval_step(self, batch):
    x, y, dvecs, name = batch
    own_dvec, target_dvec = dvecs
    y_pred = self.forward(x, target_dvec)
    return y, y_pred, target_dvec, name
```

## 11) Train model

```bash 
python src/train_model.py augmentations=augmentation_enable model=autoencoder_speaker dataset=nus_vocalset_vctk
```

* See the `conf/training/train.yaml` for more training options to override.
  * for example, append the parameter `training.batch_size=8` to change batch size
  * `training.learning_rate=0.0001` to change the learning rate
  * `training.experiment_name="experiment1"` to change the model's ckpt filename.
  * `training.max_epochs=30` to change the number of epochs to train.
  * `training.accelerator=mps` for Apple Silicon hardware

* See `conf/model/autoencoder_speaker.yaml` for model specifications to override.

## 12) Experiment Tracking
Under the `./outputs/` folder, look for the current experiment's `mlruns` folder.

e.g. `outputs/2023-03-20/20-11-30/mlruns`

In your terminal, replace the `$PROJECT_ROOT` and outputs to your full project path and run the following.

```bash 
mlflow server --backend-store-uri file:'$PROJECT_ROOT/outputs/2023-03-20/20-11-30/mlruns'
```

By default, you will be able to view the experiment tracking under `http://127.0.0.1:5000/` on your browser. 
The above is showing the configuration for MLFlow to run on localhost. 

* (Optional) it is possible to set up an MLFlow tracking server and configure the tracking uri under `training.tracking_uri`. 
See https://mlflow.org/docs/latest/tracking.html for more info.

Models will be saved into the folders as `.ckpt` under

`$PROJECT_ROOT/outputs/YYYY-MM-DD/HH-MM-SS/models`

By default, the model will save a checkpoint at every end of an epoch.

## 13) Test and Predict the model

Replace `$PATH/TO/MODEL/model.ckpt` to the saved model file, and run

```bash 
python src/test_model.py  model=autoencoder_speaker dataset=nus_vocalset_vctk testing.checkpoint_file="$PATH/TO/MODEL/model.ckpt"
```
Edit the model parameter to the model yaml file. (for this case `conf/model/autoencoder_speaker.yaml` is entered as `model=autoencoder_speaker`)

See `conf/testing/test.yaml` for more configurations.

## 14) Export trained model into ONNX format.
The script will convert the pytorch model into ONNX format, which will be needed for the plugin code.

Replace `$PATH/TO/MODEL/model.ckpt` to the saved model file,
Replace `"./models/onnx/my_model.onnx"` to specify the ONNX file path to be saved file, and run

```bash 
python src/export_model_to_onnx.py export_to_onnx.checkpoint_file="$PATH/TO/MODEL/model.ckpt" export_to_onnx.export_filename="./models/onnx/my_model.onnx"
```

Copy the ONNX file to the C++ plugin code.

# ---End of Instructions---


# Description of Scripts
- `download_data.py` -> downloads dataset into data/raw, then pick the audio and place into data/interim
- `download_pre-trained_models.py` -> download pre-trained models into models/pre-trained for later uses. 
- `process_data.py` -> use the audio from data/interim, process the audio into xx sec blocks, cuts silences and place into data/processed
- `cache_dataset.py` -> cache dataset's speech embeddings from wav files.
- `train_model.py` -> trains data from data/processed,
- `test_model.py` -> test (output as metrics) and do prediction (outputs for listening ) from data/processed
- `export_model_to_onnx.py` -> export model to onnx 