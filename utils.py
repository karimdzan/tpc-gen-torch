import torchvision.transforms as T
import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class LoadData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        # image = Image.open(os.path.join(self.root, self.classes[label], filename)).convert('RGB')
        return self.transform(image)
