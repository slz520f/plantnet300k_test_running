import os
import pandas as pd
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

class YourTestDataset(Dataset):
    def __init__(self, transform=None):
        # データセットの初期化、例えばファイルパスやラベルを読み込む
        self.data_dir = 'E:/plantnet_300K/plantnet_300K/images/new_train/test'  # テストデータのパスを指定
        self.transform = transform
        self.image_files = os.listdir(self.data_dir)  # ディレクトリ内のファイルをリスト化

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = datasets.folder.default_loader(img_path)  # 画像を読み込む
        label = ...  # ラベルを取得（必要に応じて設定）

        if self.transform:
            image = self.transform(image)

        return image, label
