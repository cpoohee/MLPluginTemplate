## python src/process_data.py process_data/dataset=nus
## python src/process_data.py process_data/dataset=vocalset
## python src/process_data.py process_data/dataset=nus_vocalset

import hydra
import librosa
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
from tqdm import tqdm
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    root_path = Path(os.path.abspath(hydra.utils.get_original_cwd()))

    # read off cfg params
    seed = cfg.process_data.seed
    ext = cfg.process_data.ext
    train_ratio = cfg.process_data.train_ratio
    val_ratio = cfg.process_data.val_ratio
    sr = cfg.process_data.sr
    clip_interval_ms = cfg.process_data.clip_interval_ms
    audio_dirs = cfg.process_data.dataset.audio_dirs
    dataset_label = cfg.process_data.dataset.dataset_label

    min_silence_len_ms = cfg.process_data.min_silence_len_ms
    silence_thresh_dbfs = cfg.process_data.silence_thresh_dbfs
    keep_silence_ms = cfg.process_data.keep_silence_ms

    print("Preparing dataset:", dataset_label)

    #train test split by speakers
    speakers = []
    for a_path in audio_dirs:
        dataset_path = Path(root_path / a_path)
        dataset_speakers = [x for x in dataset_path.iterdir() if x.is_dir()]
        speakers = speakers + dataset_speakers

    X_train, X_valtest, y_train, y_valtest = train_test_split(speakers, speakers,
                                                              test_size=(1.0 - train_ratio),
                                                              random_state=seed)
    X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest,
                                                    test_size=(1.0 - val_ratio),
                                                    random_state=seed)

    target_path = root_path / ("data/processed/" + dataset_label)

    # clear folder if exists, then make new folder
    if target_path.exists():
        print('Clearing', target_path)
        shutil.rmtree(target_path)
    Path.mkdir(target_path)

    target_path_train = target_path / 'train'
    target_path_val = target_path / 'val'
    target_path_test = target_path / 'test'
    target_path_predict = target_path / 'predict'

    process_audio(target_path_train, X_train, y_train,
                  sr=sr, audio_length_ms=clip_interval_ms, ext=ext,
                  min_silence_len_ms=min_silence_len_ms,
                  silence_thresh_dbfs=silence_thresh_dbfs,
                  keep_silence_ms=keep_silence_ms)
    process_audio(target_path_val, X_val, y_val,
                  sr=sr, audio_length_ms=clip_interval_ms, ext=ext,
                  min_silence_len_ms=min_silence_len_ms,
                  silence_thresh_dbfs=silence_thresh_dbfs,
                  keep_silence_ms=keep_silence_ms)
    process_audio(target_path_test, X_test, y_test,
                  sr=sr, audio_length_ms=clip_interval_ms, ext=ext,
                  min_silence_len_ms=min_silence_len_ms,
                  silence_thresh_dbfs=silence_thresh_dbfs,
                  keep_silence_ms=keep_silence_ms)
    process_audio(target_path_predict, X_test, y_test,
                  sr=sr, audio_length_ms=None, ext=ext)


def process_audio(target_path, audio_paths_X, audio_paths_Y,
                  sr=44100, audio_length_ms=None, ext='wav',
                  min_silence_len_ms=20,
                  silence_thresh_dbfs=-16,
                  keep_silence_ms=20):
    # delete folder if it's already there
    if target_path.exists():
        print('Clearing', target_path)
        shutil.rmtree(target_path)
    print('Creating', target_path)
    Path.mkdir(target_path)

    for speaker_path in tqdm(audio_paths_X, desc='Processing Speaker', position=0):
        # create folders for each speaker
        # WARNING: this assumes unique speaker names across different datasets!
        if (target_path/speaker_path.name).exists():
            assert False , 'speaker names are not unique!'
        Path.mkdir(target_path/speaker_path.name)

        # find audio under speaker paths
        speaker_files = librosa.util.find_files(speaker_path, ext=ext)
        for file_x in tqdm(speaker_files, desc='Processing Audio', position=1, leave=False):
            # ignore audio_paths_Y
            x = AudioSegment.from_file(file_x, ext, frame_rate=sr)

            target_path_x = (target_path / speaker_path.name)

            # force to mono
            x = x.set_channels(1)
            # peak normalization each clip
            x = effects.normalize(x)
            # fix at 16 bit... 1 is 8bit, 2 is 16 bit, 4 is 32bit. there is no 3 24bit due to api limits
            x = x.set_sample_width(4)
            if audio_length_ms is not None:
                x_silence = split_on_silence(x,
                                             # split on silences longer than xx ms
                                             min_silence_len=min_silence_len_ms,
                                             # anything under xx dBFS is considered silence
                                             silence_thresh=silence_thresh_dbfs,
                                             # keep xx ms of leading/trailing silence
                                             keep_silence=keep_silence_ms)

                if len(x_silence) == 0:
                    # do not save as it is empty.
                    continue

                # recombine
                x = AudioSegment.empty()
                for i in x_silence:
                    x += i

                # split the audio clips into 1 sec lengths, pad zero if smaller.
                x_chunks = make_chunks(x, audio_length_ms)  # Make chunks of one sec

                # Export all individual chunks as wav files
                export_chunk(x_chunks, file_x, target_path_x)
            else:
                # save to destination
                x.export(target_path_x / Path(file_x).name, format='wav')


def export_chunk(chunks, src_file, path_targ):
    """
    Helper function to save a file into chunks of file, while renaming it.
    :param chunks: audio chunks
    :param src_file: file path of the full audio
    :param path_targ: saving directory of the chunks of audio
    """
    for i, chunk in enumerate(chunks):
        chunk_name = (Path(src_file).stem + "_{0}.wav").format(i)
        chunk.export(path_targ / chunk_name, format="wav")


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


if __name__ == "__main__":
    main()
