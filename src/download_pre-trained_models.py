import os
import gdown
import urllib.request
from pathlib import Path
from tqdm import tqdm


def main():
    downlaod_archisound_ae()
    download_rtvc()
    download_vctk_ae()
    download_vf()

def downlaod_archisound_ae():
    root_path = Path(os.path.abspath(os.getcwd()))
    pre_trained_path = Path("./models/pre-trained/")
    if not pre_trained_path.exists():
        Path.mkdir(root_path / pre_trained_path)

    pre_trained_ae_path = Path("./models/pre-trained/ae")
    if not pre_trained_ae_path.exists():
        Path.mkdir(root_path / pre_trained_ae_path)

    aae_path = pre_trained_ae_path / 'pytorch_model.bin'
    aae_link = 'https://huggingface.co/archinetai/autoencoder1d-AT-v1/resolve/main/pytorch_model.bin'
    download_url(aae_link, aae_path)

    aae_conf_path = pre_trained_ae_path / 'config.json'
    aae_conf_link = 'https://huggingface.co/archinetai/autoencoder1d-AT-v1/resolve/main/config.json'
    download_url(aae_conf_link, aae_conf_path)

def download_rtvc():
    root_path = Path(os.path.abspath(os.getcwd()))
    pre_trained_path = Path("./models/pre-trained/")

    if not pre_trained_path.exists():
        Path.mkdir(root_path / pre_trained_path)

    rtvc_path = pre_trained_path / 'rt_voice_cloning_encoder.pt'

    rtvc_gdrive_link = 'https://drive.google.com/file/d/1q8mEGwCkFy23KZsinbuvdKAQLqNKbYf1/view?usp=sharing'
    gdown.download(url=rtvc_gdrive_link, quiet=False, output=rtvc_path.as_posix(), fuzzy=True)


def download_vctk_ae():
    root_path = Path(os.path.abspath(os.getcwd()))
    pre_trained_path = Path("./models/pre-trained/")
    if not pre_trained_path.exists():
        Path.mkdir(root_path / pre_trained_path)

    vctk_ae_path = pre_trained_path / 'VCTK.ts'

    vctk_ae_link = 'https://play.forum.ircam.fr/rave-vst-api/get_model/VCTK'
    download_url(vctk_ae_link, vctk_ae_path)
    return

def download_vf():
    root_path = Path(os.path.abspath(os.getcwd()))
    pre_trained_path = Path("./models/pre-trained/")

    if not pre_trained_path.exists():
        Path.mkdir(root_path / pre_trained_path)

    vf_path = pre_trained_path / 'voice_filter_embedder.pt'

    # see https://github.com/mindslab-ai/voicefilter

    vf_gdrive_link = 'https://drive.google.com/file/d/1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL/view?usp=sharing'
    gdown.download(url=vf_gdrive_link, quiet=False, output=vf_path.as_posix(), fuzzy=True)


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
