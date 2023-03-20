import hydra
import os
import librosa
import torchaudio
import pandas as pd
import torch
from pathlib import Path
import torchaudio.transforms as T
from tqdm import tqdm
from omegaconf import DictConfig
from src.model.speaker_encoder.speaker_embedder import SpeechEmbedder, AudioHelper


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    root_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))

    data_path = root_path / cfg.dataset.data_path

    embedder = SpeechEmbedder()
    chkpt_embed = torch.load(cfg.model.embedder_path, map_location=torch.device(cfg.process_data.accelerator))
    embedder.load_state_dict(chkpt_embed)
    embedder.eval()

    embedder.to(torch.device(cfg.process_data.accelerator))
    audio_helper = AudioHelper()

    # try to be as close as librosa's resampling
    resampler = T.Resample(orig_freq=cfg.dataset.block_size_speaker,
                                new_freq=embedder.get_target_sample_rate(),
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492,
                                )
    bss = cfg.dataset.block_size_speaker
    df_train = form_dataframe(data_path / 'train', resampler, audio_helper, embedder, bss)
    df_val = form_dataframe(data_path / 'val', resampler, audio_helper, embedder, bss)
    df_test = form_dataframe(data_path / 'test', resampler, audio_helper, embedder, bss)
    df_predict = form_dataframe(data_path / 'predict', resampler, audio_helper, embedder, bss)

    df_train_path = Path(data_path / 'train' / "dataframe.pkl")
    df_val_path = Path(data_path / 'val' / "dataframe.pkl")
    df_test_path = Path(data_path / 'test' / "dataframe.pkl")
    df_predict_path = Path(data_path / 'predict' / "dataframe.pkl")

    df_train.to_pickle(df_train_path)
    print('Saved to:', df_train_path)
    df_val.to_pickle(df_val_path)
    print('Saved to:', df_val_path)
    df_test.to_pickle(df_test_path)
    print('Saved to:', df_test_path)
    df_predict.to_pickle(df_predict_path)
    print('Saved to:', df_predict_path)


def form_dataframe(data_path, resampler, audio_helper, embedder, block_size_speaker):
    dataset_speakers = [x for x in data_path.iterdir() if x.is_dir()]

    speaker_names = []
    x_files = []
    dvecs = []

    for speaker in tqdm(dataset_speakers, desc='Caching speaker'):
        data_path_x = data_path / speaker.name
        speaker_files = librosa.util.find_files(data_path_x, ext='wav')
        for x_file in tqdm(speaker_files, desc='Caching audio', leave=False):
            x_files.append(x_file)
            speaker_names.append(speaker.name)

            waveform_x, _ = torchaudio.load(x_file)
            waveform_x = padding(waveform_x, block_size_speaker)
            dvec = get_embedding_vec(waveform_x, resampler, audio_helper, embedder)
            dvec = dvec.cpu().numpy()
            dvecs.append(dvec)

    data = {'x': x_files, 'speaker_name': speaker_names, 'dvec': dvecs}
    df = pd.DataFrame(data=data)
    df['related_speakers'] = ''

    # pre insert indexes of the same speakers into related_speakers
    for speaker in dataset_speakers:
        indexes_to_speaker = df[df['speaker_name'] == speaker.name].index
        df.loc[df["speaker_name"] == speaker.name, "related_speakers"] = \
            [indexes_to_speaker.tolist()]  # yes, need a nested list

    # remove 'self' index in the related_speakers
    for index, row in df.iterrows():
        indexes_to_speaker = row['related_speakers'].copy()  # somehow, we need a copy
        indexes_to_speaker.remove(index)
        df.loc[index, 'related_speakers'] = indexes_to_speaker  # no need for nested list

    return df


def get_embedding_vec(waveform_speaker, resampler, audio_helper, embedder):
    # embedding d vec
    waveform_speaker = resampler(waveform_speaker)  # resample to 16kHz
    waveform_speaker = waveform_speaker.squeeze()  # [16000]

    if torch.cuda.is_available():
        if waveform_speaker.device.type == 'cpu':
            dev = torch.device('cuda')
            waveform_speaker = waveform_speaker.to(dev)

    dvec_mel, _, _ = audio_helper.get_mel_torch(waveform_speaker)
    with torch.no_grad():
        dvec = embedder(dvec_mel)
        return dvec

def padding(waveform, target_size):
    # do padding if file is too small
    length_x = waveform.size(dim=1)
    if length_x < target_size:
        waveform = torch.nn.functional.pad(waveform,
                                           (1, target_size - length_x - 1),
                                           "constant", 0)
    return waveform


if __name__ == "__main__":
    main()
