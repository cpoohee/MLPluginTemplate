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

    print("Preparing dataset:", dataset_label)

    files = []
    for a_path in audio_dirs:
        files = librosa.util.find_files(root_path / a_path, ext=ext) + files

    # split audio clips for train validation test
    X_train, X_valtest, y_train, y_valtest = train_test_split(files, files,
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
    target_path_test_full = target_path / 'test_full'

    process_audio(target_path_train, X_train, y_train,
                  sr=sr, audio_length_ms=clip_interval_ms, ext=ext)
    process_audio(target_path_val, X_val, y_val,
                  sr=sr, audio_length_ms=clip_interval_ms, ext=ext)
    process_audio(target_path_test, X_test, y_test,
                  sr=sr, audio_length_ms=clip_interval_ms, ext=ext)
    process_audio(target_path_test_full, X_test, y_test,
                  sr=sr, audio_length_ms=None, ext=ext)


def process_audio(target_path, audio_paths_X, audio_paths_Y,
                  sr=44100, audio_length_ms=None, ext='wav'):
    # delete folder if it's already there

    if target_path.exists():
        print('Clearing', target_path)
        shutil.rmtree(target_path)
    print('Creating', target_path)

    target_path_x = target_path / 'x'
    target_path_y = target_path / 'y'

    Path.mkdir(target_path)
    Path.mkdir(target_path_x)
    Path.mkdir(target_path_y)

    audio_path_tup = merge(audio_paths_X, audio_paths_Y)
    for file_x, file_y in tqdm(audio_path_tup):
        # yes. just load both times
        x = AudioSegment.from_file(file_x, ext, frame_rate=sr)
        y = AudioSegment.from_file(file_y, ext, frame_rate=sr)

        # force to mono
        x = x.set_channels(1)
        y = y.set_channels(1)

        # peak normalization each clip
        x = effects.normalize(x)
        y = effects.normalize(y)

        # fix at 16 bit... 1 is 8bit, 2 is 16 bit, 4 is 32bit. there is no 3 24bit due to api limits
        x = x.set_sample_width(2)
        y = y.set_sample_width(2)

        if audio_length_ms is not None:
            # split the audio clips into 1 sec lengths, pad zero if smaller.
            x_chunks = make_chunks(x, audio_length_ms)  # Make chunks of one sec
            y_chunks = make_chunks(y, audio_length_ms)  # Make chunks of one sec

            # Export all individual chunks as wav files
            for i, chunk in enumerate(x_chunks):
                chunk_name = (Path(file_x).stem + "_{0}.wav").format(i)
                chunk.export(target_path_x/chunk_name, format="wav")

            # Export all individual chunks as wav files
            for i, chunk in enumerate(y_chunks):
                chunk_name = (Path(file_y).stem + "_{0}.wav").format(i)
                chunk.export(target_path_y/chunk_name, format="wav")

        else:
            # save to destination
            x.export(target_path_x / Path(file_x).name, format='wav')
            y.export(target_path_y / Path(file_y).name, format='wav')


def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


if __name__ == "__main__":
    main()