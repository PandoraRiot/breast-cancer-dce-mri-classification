
import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

LABEL_MAP = {"Benign": 0, "Malignant": 1}

DEFAULT_TRANSFORMS = T.Compose([
    T.Resize((256, 256)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])


class BreastDataset(Dataset):

    def __init__(self, csv_path, dataset_root, transforms=None, shuffle=False):
        self.dataset_root = dataset_root
        self.transforms   = transforms if transforms else DEFAULT_TRANSFORMS
        self.df           = pd.read_csv(csv_path)
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_image(self, relative_path):
        abs_path = os.path.join(self.dataset_root, relative_path)
        img      = Image.open(abs_path).convert("L")
        return self.transforms(img)

    def __getitem__(self, index):
        row   = self.df.iloc[index]
        pre   = self._load_image(row["pre_contrast"])
        early = self._load_image(row["post_early"])
        late  = self._load_image(row["post_late"])
        image = torch.cat([pre, early, late], dim=0)
        label   = LABEL_MAP[row["label"]]
        patient = row["patient"]
        return {"image": image, "label": label, "patient": patient}


class BreastLoader:

    def __init__(self, train_csv, test_csv, val_csv, dataset_root, batch_size=16, num_workers=2, transforms=None):
        self.dataset_root = dataset_root
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.transforms   = transforms
        self.train_loader = self._build_loader(train_csv, shuffle=True)
        self.test_loader  = self._build_loader(test_csv,  shuffle=False)
        self.val_loader   = self._build_loader(val_csv,   shuffle=False)
        self._print_summary()

    def _build_loader(self, csv_path, shuffle):
        dataset = BreastDataset(
            csv_path     = csv_path,
            dataset_root = self.dataset_root,
            transforms   = self.transforms,
            shuffle      = shuffle
        )
        return DataLoader(
            dataset     = dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True
        )

    def _print_summary(self):
        print("\n📦 BreastLoader inicializado:")
        print(f"  Train batches : {len(self.train_loader)}")
        print(f"  Test  batches : {len(self.test_loader)}")
        print(f"  Val   batches : {len(self.val_loader)}")
        print(f"  Batch size    : {self.batch_size}")
        print(f"  Imagen shape  : (3, 256, 256) — [pre, early, late]")
