## python src/download_data.py

import os
import shutil
import gdown
import urllib.request
import librosa.util
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def main():
    download_nus48e()
    download_vocalset()
    download_vctk()


def download_nus48e():
    ## nus 48e
    root_path = Path(os.path.abspath(os.getcwd()))
    nus_raw_path = Path("./data/raw/nus-48e")
    nus_interim_path = Path("./data/interim/nus-48e")
    nus_raw_path_posix = (root_path / nus_raw_path).as_posix()

    # Download nus48e
    if not (root_path / nus_raw_path).exists():
        print("Creating NUS48-E folder")
        Path.mkdir(root_path / nus_raw_path)
        nus_48e_link = "https://drive.google.com/drive/folders/12pP9uUl0HTVANU3IPLnumTJiRjPtVUMx"
        gdown.download_folder(url=nus_48e_link, quiet=False, output=nus_raw_path_posix)
    else:
        print("NUS48-E folder existed")

    ## move nus audio to interim folder
    if not (root_path / nus_interim_path).exists():
        print("Creating NUS48-E interim folder")
        Path.mkdir(root_path / nus_interim_path)

        for file_path in Path(nus_raw_path_posix).glob('**/*.wav'):
            # copy only singing vocals.
            if 'sing' in str(file_path):
                target_path = root_path / nus_interim_path / file_path.name
                i = 1
                while True:
                    if target_path.exists():
                        target_path = root_path / nus_interim_path / (
                                file_path.stem + '_' + str(i) + file_path.suffix)
                        i += 1
                        continue
                    else:
                        break
                shutil.copy(file_path, target_path)
    else:
        print("NUS48-E interim folder existed")


def download_vocalset():
    root_path = Path(os.path.abspath(os.getcwd()))
    ## vocalset
    vocalset_raw_path = Path("./data/raw/vocalset")
    vocalset_interim_path = Path("./data/interim/vocalset")
    vocalset_raw_path_posix = (root_path / vocalset_raw_path).as_posix()
    vocalset_filename = "VocalSet11.zip"

    # download vocalset
    if not (root_path / vocalset_raw_path).exists():
        print("Creating vocalset folder")
        Path.mkdir(root_path / vocalset_raw_path)

        ## use v1.1 version for smaller download than v1.2
        vocalset_link = "https://zenodo.org/record/1203819/files/VocalSet11.zip?download=1"
        download_url(vocalset_link, (root_path / vocalset_raw_path / vocalset_filename).as_posix())
    else:
        print("vocalset folder existed")

    ## move vocalset folder to interim
    if not (root_path / vocalset_interim_path).exists():
        print("Creating vocalset interim folder")
        Path.mkdir(root_path / vocalset_interim_path)

        # extract zip file
        shutil.unpack_archive(root_path / vocalset_raw_path / vocalset_filename,
                              root_path / vocalset_raw_path)

        # copy extracted wav to interim
        for file_path in Path(vocalset_raw_path_posix).glob('**/*.wav'):
            # include this dataset's spoken files
            target_path = root_path / vocalset_interim_path / file_path.name
            i = 1
            while True:
                if target_path.exists():
                    target_path = root_path / vocalset_interim_path / (
                            file_path.stem + '_' + str(i) + file_path.suffix)
                    i += 1
                    continue
                else:
                    break
            shutil.copy(file_path, target_path)

        # delete original extracted files
        shutil.rmtree(root_path / vocalset_raw_path / '__MACOSX');
        shutil.rmtree(root_path / vocalset_raw_path / 'FULL');
        (root_path / vocalset_raw_path / 'readme-anon.txt').unlink()
        (root_path / vocalset_raw_path / 'test_singers_technique.txt').unlink()
        (root_path / vocalset_raw_path / 'train_singers_technique.txt').unlink()
        (root_path / vocalset_raw_path / 'DataSetVocalises.pdf').unlink()
    else:
        print("vocalset interim folder existed")


def download_vctk():
    root_path = Path(os.path.abspath(os.getcwd()))

    ## vctk
    vctk_raw_path = Path("./data/raw/vctk")
    vctk_interim_path = Path("./data/interim/vctk")
    vctk_raw_path_posix = (root_path / vctk_raw_path).as_posix()
    vctk_filename = "DS_10283_3443.zip"

    # download vctk
    if not (root_path / vctk_raw_path).exists():
        print("Creating vctk folder")
        Path.mkdir(root_path / vctk_raw_path)
        vctk_link = 'https://datashare.ed.ac.uk/download/DS_10283_3443.zip'
        download_url(vctk_link, (root_path / vctk_raw_path / vctk_filename).as_posix())
    else:
        print("vctk folder existed")

    ## move vctk folder to interim
    if not (root_path / vctk_interim_path).exists():
        print("Creating vctk interim folder")
        Path.mkdir(root_path / vctk_interim_path)

        vctk_raw_zip_path = root_path / vctk_raw_path / Path("./VCTK-Corpus-0.92.zip")
        if not vctk_raw_zip_path.exists():
            # extract zip file
            shutil.unpack_archive(root_path / vctk_raw_path / vctk_filename,
                                  root_path / vctk_raw_path)

        if not (root_path / vctk_raw_path / Path('./wav48_silence_trimmed')).exists():
            shutil.unpack_archive(vctk_raw_zip_path,
                                  root_path / vctk_raw_path)

        flac_files = librosa.util.find_files(root_path / vctk_raw_path, ext='flac')

        # copy extracted flac to wav to interim
        for file_path in tqdm(flac_files):
            # send all of this dataset's spoken files
            file_path = Path (file_path)
            target_path = root_path / vctk_interim_path / file_path.name
            i = 1
            while True:
                if target_path.exists():
                    target_path = root_path / vctk_interim_path / (
                            file_path.stem + '_' + str(i) + file_path.suffix)
                    i += 1
                    continue
                else:
                    break
            data, sr = librosa.load(file_path)
            sf.write(target_path.with_suffix('.wav'), data, sr, subtype='PCM_16')

        # delete original extracted files
        shutil.rmtree(root_path / vctk_raw_path / 'txt');
        shutil.rmtree(root_path / vctk_raw_path / 'wav48_silence_trimmed');
        (root_path / vctk_raw_path / 'license_text').unlink()
        (root_path / vctk_raw_path / 'README.txt').unlink()
        (root_path / vctk_raw_path / 'speaker-info.txt').unlink()
        (root_path / vctk_raw_path / 'update.txt').unlink()
        (root_path / vctk_raw_path / 'VCTK-Corpus-0.92.zip').unlink()

    else:
        print("vctk interim folder existed")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if __name__ == "__main__":
    main()
