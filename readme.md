# Introduction
This project investigates the use of AI generative models for music production.
This was intended to be a short project aimed for submission for [Neural Audio Plugin Competition 2023](https://www.theaudioprogrammer.com/neural-audio).

However, due to lack of time, the model is not working at the moment. 
Therefore, this repo will be repurposed to be a template for future models to be explored.

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

## 2) Major Libraries Used for Development
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

## 5) Download Dataset

A script is created for downloading the following datasets
- NUS-48E
- VocalSet
- VCTK

Go to the project root folder

```bash 
cd MLPluginTemplate
```

Run the download script
```bash 
python src/download_data.py
``` 

* in case of failure especially from gdrive links, you try to delete contents of `/data/raw` and try again later. 
the scripts will skip the download if it detects the folder exists.

The script will download the raw datasets and saves it onto `/data/raw` folder.
It also transcodes to `wav` format and transfers useful audio files to the `/data/interim` folder.

## 6) Download Pre-trained Models
Run the download script

```bash 
python src/download_pre-trained_models.py 
```

It will download models into the `/models/pre-trained` folder

## 7) Pre-process Dataset
Continue running from the same project root folder,

```bash 
python src/process_data.py process_data/dataset=nus_vocalset_vctk
```

The above command will pre-process NUS-48E, VCTK and vocalset datasets.

Pre-processing will split the dataset into train/validation/test/prediction splits as stored into the `./data/processed` folder.
It also slices the audio into 5 sec long clips.

### (Optional) replace it with options:

* `process_data/dataset=nus` for pre-processing just the NUS-48E dataset
* `process_data/dataset=vctk` for pre-processing just the VCTK dataset

* See the `conf/process_data/process_root.yaml` for more detailed configurations.

## 8) Cache speech encodings
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

## 9) Train model 
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

## 10) Experiment Tracking
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

## 11) Test and Predict the model

Replace `$PATH/TO/MODEL/model.ckpt` to the saved model file, and run

```bash 
python src/test_model.py  model=autoencoder_speaker dataset=nus_vocalset_vctk testing.checkpoint_file="$PATH/TO/MODEL/model.ckpt"
```

## 12) Export trained model into ONNX format.
The script will convert the pytorch model into ONNX format, which will be needed for the plugin code.

Replace `$PATH/TO/MODEL/model.ckpt` to the saved model file,
Replace `"./models/onnx/my_model.onnx"` to specify the ONNX file path to be saved file, and run

```bash 
python src/export_model_to_onnx.py export_to_onnx.checkpoint_file="$PATH/TO/MODEL/model.ckpt" export_to_onnx.export_filename="./models/onnx/my_model.onnx"
```

Copy the ONNX file to the C++ plugin code.

# ---End of Instructions---


# Background
After some literature review, some possibilities in audio AI that are suitable for music production are:

- Singing/Speech Voice Conversion (VC)
- Singing/Speech Style Conversion
- Text to Speech (TTS)
- Singing Voice Synthesis
- Audio source separation

There are definitely more AI applications than listed above.

However, most of the demos are lacking in fidelity with audio downsampled to 20khz or less. 
Ideally it should be is at least 44100Hz.
The resultant generated audio are therefore missing high-end frequencies 10khz and more.
Our plugin should avoid sacrifice fidelity. 

# Practical Plugin Goals
- To generate high quality audio usable for mixing. (sample rate >= 44100Hz)
- plugin latency should be low. ( <= 1 sec of samples )
- deterministic/reproducibility of plugin models ( the AI model should not churn out a different output for the same playback)
- Model size should be acceptable for plugin installation. ( <200MB )
- CPU usage should also be acceptable. (10 instances running real time in a DAW at the same time)

The end result should produce natural sounding audio and also subjectively appealing.

## Double tracking Ideas
In a typical highly produced multitrack for mixing, the vocal double/triples or more could be recorded for mixing. 
An experienced mixer will be able to utilise the doubling effect to enhance the performance by adding the double track balanced just below the lead vocal track.
The resulting vocal performance will cut through the mix and sound thicker.

A simple copy of the same track does not work as doubling as the sum of two identical track just results in a 3db louder audio. 
Therefore, a double track is always taken from a different take. 
The differences in (and not limited to) phase, pitch, timing, timbre of a fresh take all contributes to the doubling effect.

It is even more so for background vocals which are usually produced with more than 2 takes of the same parts, multiplied by the harmony lines.

Without double tracks, a mixer do make use of some existing artificial techniques that mimic doubling. 
For example, de-tuning, delaying, chorusing a copy the same track. See Waves's doubler.
Given an option to a mixer however, it is likely we will choose the real double take over the synthetic doubler.   

Few have approached the generative audio for producing double takes that is suitable for the audio doubling effect. 
For a mixer, this is a potential time and money saver.

It is also hoped that using vocal conversion, it will produce a timbre based doubling effect sounding natural enough for the listener.  

Some papers that might be related are: 

```bibtex
@inproceedings{tae2021mlp,
  title={Mlp singer: Towards rapid parallel korean singing voice synthesis},
  author={Tae, Jaesung and Kim, Hyeongju and Lee, Younggun},
  booktitle={2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```

```bibtex
@article{tamaru2020generative,
  title={Generative moment matching network-based neural double-tracking for synthesized and natural singing voices},
  author={Tamaru, Hiroki and Saito, Yuki and Takamichi, Shinnosuke and Koriyama, Tomoki and Saruwatari, Hiroshi},
  journal={IEICE TRANSACTIONS on Information and Systems},
  volume={103},
  number={3},
  pages={639--647},
  year={2020},
  publisher={The Institute of Electronics, Information and Communication Engineers}
}
```

```bibtex
@inproceedings{albadawy2020voice,
  title={Voice Conversion Using Speech-to-Speech Neuro-Style Transfer.},
  author={AlBadawy, Ehab A and Lyu, Siwei},
  booktitle={Interspeech},
  pages={4726--4730},
  year={2020}
}
```

```bibtex
@inproceedings{qian2019autovc,
  title={Autovc: Zero-shot voice style transfer with only autoencoder loss},
  author={Qian, Kaizhi and Zhang, Yang and Chang, Shiyu and Yang, Xuesong and Hasegawa-Johnson, Mark},
  booktitle={International Conference on Machine Learning},
  pages={5210--5219},
  year={2019},
  organization={PMLR}
}
```

## Ideas tried
One way is to utilise existing pre-trained speaker encoder from https://github.com/mindslab-ai/voicefilter.
This is where X -> speaker encoder -> S, where X is a waveform, S is a speaker embedding vector of 256. 
The speaker embedder was originally used to classify and identify 3549 speakers from libriSpeech and VoxCeleb1 dataset. 
Therefore, it can generate embeddings that contain latent variables representative of a speaker's voice. 
The AutoVC also used a pre-trained speaker encoder for their Voice conversion model.   

We investigated Real-time Voice Cloning (RTVC) encoder and Voice Filter's (VF) encoder. 
In our notebook `investigate_speaker_embedding_RTVC.ipynb` and `investigate_speaker_embedding_VF.ipynb`, 
we show that VF encoder is likely to be more performant using the t-SNE plot.  

Next we looked for a pre-trained AutoEncoder for generative audio. 
See our notebook `investigate_pre-trained_autoencoder.ipynb` ,`investigate_rave_autoencoder.ipynb`.
Our brief inference in the notebook studying the sound generated from Rave's Auto Encoder and Archisound https://github.com/archinetai/archisound shows that using Archisound's 
`autoencoder1d-AT-v1 Reconstruction` model could sound better for our singing voice data.

Studying further on the `autoencoder1d-AT-v1 Reconstruction` bottlenecks, we have findings shown in notebooks
`reducing_bottleneck_of_AE(channels).ipynb` and `reducing_bottleneck_of_AE.ipynb`. 
It shows how much audio information is being represented in the latent z bottleneck vectors. 

We let the current speaker input to the AE as y. 
Another target speaker's voice as y_target.

For the AE to encode and decode, we let 

` z = AE.encode(y)`

` y' = AE.decode(z)`

where `y'`is the reconstructed audio from the AE.
The current z vector has 32 channels followed by samples//32 sized.  

To get speaker embedding, we call

`dvec_target = speakerEmbed(y_target)`

note that the dvec is a 256 sized vector.

We attempted an LSTM layer and a SALN layer that tries to learn and encode the latent variables with speaker embeddings `dvec`

- `z' = LSTM(concat [dvec, z])` , or
- `z' = StyleAdaptiveLayerNorm(z, dvec)`

which is then passed to the AE decode `y' = AE.decode(z')` 

Loss functions were:

`loss = lambda_wav * mse(y, y') + lambda_emb * mse(dvec_target, speakerEmbed(y'))` 

which is a weighted regression mse of waveform, and mse of the speaker embeddings.

The experiments setup were to freeze the AE weights. and train the LSTM or SALN layer.

For LSTM,
First 20 epochs were conducted, under those conditions: 

- frozen AE weights. 
- input y and target y belongs to the same speaker to test perfect reconstruction.
- lambda_emb set to 0

Results listening test from the initial 20 epoch LSTM, was resulting in very noisy sounds, nothing like a perfect reconstruction.

Therefore, this idea was abandoned. We try the simpler SALN, which was adatped from meta-stylespeech.

```bibtex
@inproceedings{min2021meta,
  title={Meta-stylespeech: Multi-speaker adaptive text-to-speech generation},
  author={Min, Dongchan and Lee, Dong Bok and Yang, Eunho and Hwang, Sung Ju},
  booktitle={International Conference on Machine Learning},
  pages={7748--7759},
  year={2021},
  organization={PMLR}
}
```

For SALN, 
first 20 epochs were conducted, under those conditions: 

- frozen AE weights. 
- input y and target y belongs to the same speaker to test perfect reconstruction.
- lambda_emb set to 0
- learning rate at 1e-4

results was a good reconstruction.

We continued training the model until 40 epochs with:
- frozen AE encoder weights 
- unfrozen AE decoder
- lambda_emb set to 10
- learning rate at 1e-4

The result was a repeated noise, which it might be trying to match a noisy sound that lowers the MSE of embedding but however it sounds horrible.

We rolled back and trained from the 20 epoch checkpoint to 40 epochs with 
- frozen AE weights. 
- lambda_emb and lambda_wav set to 1
- learning rate at 1e-4

The result was low quality reconstructed playback even though MSE losses were low.

Then we continued the next 20 epochs with 
- frozen AE encoder weights 
- unfrozen AE decoder
- lambda_emb and lambda_wav set to 1
- lowered learning rate to 1e-5

The result was low quality reconstructed playback too. And this is where I ran out of time. 

## Learning outcomes
After thoughts, the current approach to infuse latent embeddings are limited to the alignment of the audio. 
Each embedding vector only has an influence to a fixed blocksize sliced from a continuous audio. 
This is the result of using a time domain AE. 

If the input is in frequency domain, it could potentially be better.

Some kind of flexible time shifting and alignments mechanism are need to consider for phase shifts to work. 
The current model is not able to do that.

We might need some kind of discriminator network for our loss objective function. 
The current MSE loss on embeddings might on the surface learns to closely match dvecs, but by doing so produces noise without meaningful audio. 

# Other Findings
- multi res stft loss function creates the most natural sounding generation, tested on basic wavenet reproduction training.
- melspectrum and any preemphasis also resulted in less natural sounding generation (also tested on basic wavenet reproduction)
- augmentations done on training/testing data only produces models that predicts the original wav, 
  - the resulting audio is almost a copy of input.
  - might need a new loss function to penalise exact copy. even then, the audio could be phased flip, eg cossimloss == 1 or -1 
- realisation that for any useful effects to be used, training from scratch is not practical for this competition.
  - current machine is not capable to run experiments in time.
  - need to find pre-trained models, fine-tune and adapt to other potential useful plugins. 
- pre-trained models also suffer reconstruction artifacts.
  - there is an issue of degrading audio from short sample blocks, any lower than 32768 samples. (nearly 0.68 sec)
  - we should be able to oversample the audio, passing 32768 samples into the pre-trained model(trained in 48kHz), which represents a shorter blocktime (0.17 sec).
  - might explore `cached_conv` library to solve clicks from inferencing the beginning of the sample block. Onnx might not be able to convert it??
  - the 32 channels in the bottleneck vector z, where it is sized [batch, 32, T], is likely to represent some frequency bands based on the quick experiment on zeroing out some channels. See (notebooks/reducing_bottleneck_of_AE(channels).ipynb) 


## Future AI Ideas

- lead vocal separation from other voices. 
  - Useful for recordings done with multiple singers in the same room. 
  - E.g. 
    - the need to keep the lead vocal, but the harmony from second singer is out of tune.
    - say 2 singers sang in unison, but production needs a harmony instead. potential autotune application.
- Search for more pre-trained models suitable for fine-tuning. 

# Datasets
- NUS-48E
  - [Duan, Zhiyan, et al. "The NUS sung and spoken lyrics corpus: A quantitative comparison of singing and speech." 2013 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference. IEEE, 2013.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6694316)
  - 12 singers with 4 songs for each singer
  - 48 pairs of sung and spoken
- VocalSet
  - [Wilkins, Julia, et al. "VocalSet: A Singing Voice Dataset." ISMIR. 2018.](https://zenodo.org/record/1193957#.Y_zvvexBzA0)

- VCTK
  - [Veaux, Christophe, Junichi Yamagishi, and Kirsten MacDonald. "CSTR VCTK corpus: English multi-speaker corpus for CSTR voice cloning toolkit." University of Edinburgh. The Centre for Speech Technology Research (CSTR) (2017).](https://datashare.ed.ac.uk/handle/10283/3443)


# Description of Scripts
- `download_data.py` -> downloads dataset into data/raw, then pick the audio and place into data/interim
- `download_pre-trained_models.py` -> download pre-trained models into models/pre-trained for later uses. 
- `process_data.py` -> use the audio from data/interim, process the audio into xx sec blocks, cuts silences and place into data/processed
- `cache_dataset.py` -> cache dataset's speech embeddings from wav files.
- `train_model.py` -> trains data from data/processed,
- `test_model.py` -> test (output as metrics) and do prediction (outputs for listening ) from data/processed
- `export_model_to_onnx.py` -> export model to onnx 