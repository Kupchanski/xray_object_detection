import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from .dataset import XrayDataset, VinXrayDataset
from .transforms import get_train_transforms, get_val_transforms


class XrayData(pl.LightningDataModule):
    def __init__(self, 
                xray_dir,
                vin_dir,
                annotations_root,
                batch_size,
                image_size = 512,
                only_xray = False,
                ):
        """Xray datamodule

        Args:
            xray_dir (_type_): xray dataset images folder path
            vin_dir (_type_): vinbigdata chest dataset images folder path
            annotations_root (_type_): csv files with labels folder path
            batch_size (_type_): batch size for train/validation
            image_size (int, optional): Defaults to 512.
            only_xray (bool, optional): Make True for only xray dataset training. Defaults to False.
        """

        super().__init__()
        self.xray_dir = xray_dir
        self.vin_dir = vin_dir
        self.bs = batch_size
        self.annotations_root = annotations_root
        self.only_xray = only_xray
        self.image_size = image_size

    # init all datasets
    def get_dataset(self):
        self.train_dataset = XrayDataset(
            root=self.xray_dir,
            annotations_file=os.path.join(self.annotations_root, 'train_df.csv'),
            transforms=get_train_transforms(self.image_size),
        )

        self.vin_train_dataset = VinXrayDataset(
            root=self.vin_dir,
            annotations_file=os.path.join(self.annotations_root, 'vin_train_df.csv'),
            transforms=get_train_transforms(self.image_size),
        )

        self.val_dataset = XrayDataset(
            root=self.xray_dir,
            annotations_file=os.path.join(self.annotations_root, 'val_df.csv'),
            transforms=get_val_transforms(self.image_size),
        )
        self.vin_val_dataset = VinXrayDataset(
            root=self.vin_dir,
            annotations_file=os.path.join(self.annotations_root, 'vin_val_df.csv'),
            transforms=get_val_transforms(self.image_size),
        )
        # if only_xray param - False, concat 2 datasets to train together
        if self.only_xray:
            self.all_train = self.train_dataset
            self.all_val = self.val_dataset
        else:
            self.all_train = ConcatDataset([self.train_dataset, self.vin_train_dataset])
            self.all_val = ConcatDataset([self.val_dataset, self.vin_val_dataset])



    def train_dataloader(self):
        return DataLoader(
            self.all_train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.all_val,
            batch_size=self.bs,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )


