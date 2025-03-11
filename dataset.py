from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class dataset(Dataset):
    def __init__(self, df, data_path, transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.data_dir = 'train_images'

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        img_name, label = self.df_data[item]
        img_path = os.path.join(self.data_path, self.data_dir, img_name)
        # 读取为numpy数组以适配albumentations
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transforms is not None:
            transformed = self.transforms(image=img)
            image = transformed["image"]
            return image, label
        return img, label
