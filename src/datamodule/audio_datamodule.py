import pytorch_lightning as pl
import librosa
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from src.datamodule.audio_dataloader import AudioDataset
from src.datamodule.audio_dataloader_pred import AudioDatasetPred
from torch.utils.data import DataLoader

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path,
                 cfg: DictConfig,
                 batch_size: int = 32,
                 shuffle_train:bool = True,
                 do_aug_in_predict: bool = False,
                 do_aug_in_val: bool = False,
                 do_aug_in_test: bool = False,
                 do_aug_in_train: bool = True,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.ext = cfg.process_data.ext
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.num_workers = cfg.training.num_workers
        self.do_aug_in_predict = do_aug_in_predict
        self.do_aug_in_val = do_aug_in_val
        self.do_aug_in_test = do_aug_in_test
        self.do_aug_in_train = do_aug_in_train
        self.shuffle_train = shuffle_train
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == "fit":
            self.df_train = self.form_dataframe(self.data_dir / 'train')
            self.df_val = self.form_dataframe(self.data_dir / 'val')
        if stage == "test":
            self.df_test = self.form_dataframe(self.data_dir / 'test')
        if stage == "predict":
            self.df_predict = self.form_dataframe(self.data_dir / 'predict')

    def form_dataframe(self, data_path):
        dataset_speakers = [x for x in data_path.iterdir() if x.is_dir()]

        speaker_names = []
        x_files = []

        for speaker in dataset_speakers:
            data_path_x = data_path / speaker.name
            speaker_files = librosa.util.find_files(data_path_x, ext=self.ext)
            for x_file in speaker_files:
                x_files.append(x_file)
                speaker_names.append(speaker.name)

        data = {'x': x_files, 'speaker_name': speaker_names}
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

    def train_dataloader(self):
        assert (self.df_train is not None)
        train_set = AudioDataset(self.df_train,
                                 cfg=self.cfg,
                                 do_augmentation=self.do_aug_in_train)
        persist_worker = True if self.num_workers > 0 else False
        return DataLoader(train_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=persist_worker,
                          shuffle=self.shuffle_train)

    def val_dataloader(self):
        assert (self.df_val is not None)
        val_set = AudioDataset(self.df_val, cfg=self.cfg, do_augmentation=self.do_aug_in_val)
        persist_worker = True if self.num_workers > 0 else False

        return DataLoader(val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=persist_worker)

    def test_dataloader(self):
        assert (self.df_test is not None)
        test_set = AudioDataset(self.df_test, cfg=self.cfg, do_augmentation=self.do_aug_in_test)
        persist_worker = True if self.num_workers > 0 else False
        return DataLoader(test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=persist_worker)

    def predict_dataloader(self):
        assert (self.df_predict is not None)
        if self.do_aug_in_predict:
            pred_set = AudioDataset(self.df_predict, cfg=self.cfg, do_augmentation=True)
            pred_set.set_random_crop(False)
        else:
            pred_set = AudioDatasetPred(self.df_predict, cfg=self.cfg)
        return DataLoader(pred_set, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
