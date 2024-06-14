import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

# some augmenantions 

def get_train_transforms(img_size=512):
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.RandomScale(p=0.8),
            albu.Resize(height=img_size, width=img_size, p=1),
            albu.OneOf([
                albu.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                albu.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.5),
            albu.ChannelDropout(p=0.5),
            albu.OneOf(
                    [
                        albu.MotionBlur(p=0.5),
                        albu.MedianBlur(p=0.5),
                        albu.GaussianBlur(p=0.5),
                        albu.GaussNoise(p=0.5),
                    ],
                    p=0.5,
                ),
            albu.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            #albu.Normalize(),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=albu.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_val_transforms(img_size=512):
    return albu.Compose(
        [
            albu.Resize(height=img_size, width=img_size, p=1),
            #albu.Normalize(),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=albu.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )