import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=640):
    # Standard augmentation pipeline for industrial defect images
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def get_val_transforms(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
