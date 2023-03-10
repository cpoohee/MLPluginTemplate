# Introduction
This project investigates the use of AI generative models create vocal doubles for mixing from a single vocal take.
This is intended to be a short project aimed for submission for [Neural Audio Plugin Competition](https://www.theaudioprogrammer.com/neural-audio).

# Installation Instructions
- placeholder

# Background
Some literature review on what's possible in recent audio AI, and suitable for music production are:

- audio source separation
- Singing/Speech Voice Conversion (VC)
- Singing/Speech Style Conversion
- Text to Speech (TTS)
- Singing Voice Synthesis 

There are definitely more AI applications than listed above.

Most of the demos are lacking in fidelity with audio downsampled to 20khz or less. 
The resultant generated audio are therefore missing high-end frequencies 10khz and more.
Our plugin should avoid that sacrifice fidelity. 

Personally, I have experience in recording and mixing music too. 
In a live performance recording with a small group of musicians (5 people), it is difficult to get a refined vocal lead/background performances comparable to highly produced production.

In a typical highly produced multitrack for mixing, the vocal double/triples or more could be recorded for mixing. 
An experienced mixer will be able to utilise the doubling effect to enhance the performance by adding the double track balanced just below the lead track.
The resulting vocal performance will cut through the mix and sound thicker.

A simple copy of the same track does not work as doubling as the sum of two identical track just results in a 3db louder audio. 
Therefore, a double track is always a different take from the lead track. 
The differences in (and not limited to) phase, pitch, timing, tone of a fresh take all contributes to the doubling effect.

Even more so for background vocals, they are usually more than 2 takes of the same parts, multiplied by the harmony lines.

Without double tracks, a mixer do make use of some existing artificial techniques that mimic doubling. 
For example, de-tuning, delaying, chorusing a copy the same track. See Waves's doubler.
However, a mixer will want to have the option to choose the real double take over the synthetic doubler.   

Few have approached the generative audio from the same audio to produce a double take that is suitable for the audio doubling effect. 
Probably it is not exciting enough to publish a research work, but for a mixer, this is a potential time and money saver.
It is also hoped that with enough variations and stacked layers of this plugin, it will sound natural enough for the listener.  

Some papers that might be related are: 

- [Tae, Jaesung, Hyeongju Kim, and Younggun Lee. "Mlp singer: Towards rapid parallel korean singing voice synthesis." 2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2021.](https://arxiv.org/abs/2106.07886)
- [Tamaru, Hiroki, et al. "Generative moment matching network-based neural double-tracking for synthesized and natural singing voices." IEICE TRANSACTIONS on Information and Systems 103.3 (2020): 639-647.](https://www.jstage.jst.go.jp/article/transinf/E103.D/3/E103.D_2019EDP7228/_pdf)
- [AlBadawy, Ehab A., and Siwei Lyu. "Voice Conversion Using Speech-to-Speech Neuro-Style Transfer." Interspeech. 2020.](https://ebadawy.github.io/post/speech_style_transfer/Albadawy_et_al-2020-INTERSPEECH.pdf)
- [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/pdf/1905.05879.pdf)
- 
For ML datasets, usually we feed an input X and target Y for the model to learn (X->Y).

Potentially, our dataset for double takes could be easier if the model learns by using the same input audio as the target.
- (X -> X).

The model will therefore predict X* from the input X.
- (X->X*).

Our (X - X*) should be non-zero but close enough to be a good double. 
This will be subjected to some measurable metrics for objective comparison. 

If we are able to find double take datasets, it will be useful as well. 
That is X is the lead, Y is the double.

The end result should produce natural sounding audio which might only be subjectively judged.

Lastly, the model should be small and performant enough to run in a plugin.

# Goal
- To generate high quality audio usable for mixing. (sample rate >= 44100Hz)
- plugin latency should be low. ( <= 1 sec of samples )
- deterministic/reproducibility of plugin models ( the AI model should not churn out a different output for the same playback)
- Model size should be acceptable for plugin installation. ( <200MB )
- CPU usage should also be acceptable. (10 instances running real time in a DAW at the same time)

# Stretched Goal
- provide a vocal personality for VC, and use it for background harmony.
- auto generate harmony similar to Waves's Harmony plugin, with our vocal personality.

# ToDos
- Create a simple JUCE plugin without AI as a start. (done)
- Search good quality datasets, do preprocessing (done)
- prepare possible augmentations (done)
- create the basic wavenet (done)
- investigate loss functions (done)
  - checked out ESR, DC, LogCosh, SNR, SDSDR, MSE, various stft 
  - quick training suggest each have its own weakness see results.xlsx in models
  - sticking to multi res stft after it sounds the most natural 
- train a decent model without bells and whistles, etc augmentation. just able to produce identity sound will do. (done)
- create a pipeline of model deployment into JUCE (done)
  - convert model to ONNX, 
  - in JUCE, use ONNX runtime in c++
- Iterate experiments. 
  - check out STFT based loss functions, already part of auraloss (done)
  - check U wave net.
  - train 'spectral recovery' type of neural model
    - using low passed input. predict full spectrum sound.
  - Increase model size on the small NUS dataset, evaluate usefulness
  - Full dataset training on finalised model
  - Try voice conversion.. that worked with lower sampled sounds.
    - train vae gan https://github.com/ebadawy/voice_conversion,
    - use pretrained logmel vocoder  https://github.com/r9y9/wavenet_vocoder 
    - use our spectal recovery model to recover highend sounds.
- Improve plugin usefulness
  - UI
  - Sound, might need tricks to beat the weird bleep during the front of output 
- prepare a video demo

# Stretched ToDos
- download and extract free multitracks that contains vocal doubles, use it to train/ fine-tune the model

# Datasets
- NUS-48E
  - [Duan, Zhiyan, et al. "The NUS sung and spoken lyrics corpus: A quantitative comparison of singing and speech." 2013 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference. IEEE, 2013.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6694316)
  - 12 singers with 4 songs for each singer
  - 48 pairs of sung and spoken
- VocalSet
  - [Wilkins, Julia, et al. "VocalSet: A Singing Voice Dataset." ISMIR. 2018.](https://zenodo.org/record/1193957#.Y_zvvexBzA0)

- VCTK
  - [Veaux, Christophe, Junichi Yamagishi, and Kirsten MacDonald. "CSTR VCTK corpus: English multi-speaker corpus for CSTR voice cloning toolkit." University of Edinburgh. The Centre for Speech Technology Research (CSTR) (2017).](https://datashare.ed.ac.uk/handle/10283/3443)
- more data to come...

# Experiments
- use a simple wavenet. time based input/output, convert to run in JUCE
- use a STFT-based input to investigate latency limitation or benefits in natural sounding
- Decide on method and improve.

# Findings
- multi res stft loss function creates the most natural sounding generation, tested on basic wavenet
- melspectrum, any preemphasis also resulted in less natural sounding generation
- augmentations done on training/testing data only produces models that predicts the original wav, 
  - the resulting audio is almost a copy of input.
  - might need a new loss function to penalise exact copy. even then, the audio could be phased flip, eg cossimloss == 1 or -1 
- realisation that for any useful effects to be used, training from scratch is not practical for this competition.
  - current machine is not capable to run experiments in time.
  - need to find pre-trained models, fine-tune and adapt to other potential useful plugins. 
# Brief description of Source code folder and scripts
- download_data.py -> downloads dataset into data/raw, then pick the audio and place into data/interim
- process_data.py -> use the audio from data/interim, process the audio into 1 sec blocks, cuts silences and place into data/processed
- train_wavenet.py -> trains data from data/processed,
- train_wavenet.py -> test (output as metrics) and do prediction (outputs for listening ) from data/processed
- export_model_to_onnx.py -> export model to onnx 