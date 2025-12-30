import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FER2013(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Mapping labels as per your notebook
        # 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
        split_dir = os.path.join(root, split)
        for label, emotion in enumerate(sorted(os.listdir(split_dir))):
            emotion_dir = os.path.join(split_dir, emotion)
            for img_name in os.listdir(emotion_dir):
                self.images.append(os.path.join(emotion_dir, img_name))
                self.labels.append(label)

    def __getitem__(self, index):
        path = self.images[index]
        label = self.labels[index]
        img = Image.open(path).convert('RGB') # Models expect 3 channels for transfer learning
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

def get_transforms():
    # Training uses RandomCrop(44) as mentioned in your paper [cite: 125, 184]
    train_transform = transforms.Compose([
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # Testing uses TenCrop(44) for better robustness [cite: 125, 184]
    test_transform = transforms.Compose([
        transforms.TenCrop(44),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    return train_transform, test_transform