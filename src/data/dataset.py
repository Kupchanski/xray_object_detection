import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

from .utils import read_rgb


class XrayDataset(Dataset):
    def __init__(self, root, annotations_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotations_df = pd.read_csv(annotations_file, index_col=[0])

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        image_id = row['Image Index']
        img_path = os.path.join(self.root, image_id)
        image = read_rgb(img_path)
        # to pascal vos format
        box = [
            row['x'],
            row['y'],
            row['x'] + row['w'],
            row['y'] + row['h'],
        ]
        label = row['label']
        box = np.array(box)

        # apply augmentations
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=[box], labels=[label])
            image = transformed['image']
            _, img_h, img_w = image.shape
            bboxes = transformed['bboxes']
            labels = transformed['labels']
            # make dict of the features for EffDet training.
            target = {
                'bbox': torch.as_tensor(bboxes, dtype=torch.float32),
                'cls': torch.as_tensor(labels, dtype=torch.float32),
                'img_size': torch.tensor([img_h, img_w]),
                'img_scale': torch.tensor([1.0]),
            }
        return image / 255., target


# same dataset for VinBigData dataset. Have some minor differences in data format
class VinXrayDataset(Dataset):
    def __init__(self, root, annotations_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotations_df = pd.read_csv(annotations_file, index_col=[0])

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        image_id = row['image_id']
        image_file_name = image_id + ".png"
        img_path = os.path.join(self.root, image_file_name)
        image = read_rgb(img_path)
        # to pascal vos format
        box = [
            row['x_min'],
            row['y_min'],
            row['x_max'],
            row['y_max'],
        ]

        label = row['class_id']
        box = np.array(box)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=[box], labels=[label])
            image = transformed['image']
            _, img_h, img_w = image.shape
            bboxes = transformed['bboxes']
            labels = transformed['labels']
            target = {
                'bbox': torch.as_tensor(bboxes, dtype=torch.float32),
                'cls': torch.as_tensor(labels, dtype=torch.float32),
                'img_size': torch.tensor([img_h, img_w]),
                'img_scale': torch.tensor([1.0]),
            }

        return image / 255. , target
