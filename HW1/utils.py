import os
from PIL import Image
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        img_name = os.path.splitext(self.image_files[idx])[0]
        if self.transform:
            image = self.transform(image)
        return image, img_name


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from train import MyResNet
    model = MyResNet()
    print(count_trainable_parameters(model))
