import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms


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


class MyResNet(nn.Module):
    def __init__(self, num_classes=100, num_hidden=0, freeze_pretrained=False):
        super(MyResNet, self).__init__()
        self.resnet = models.resnet101(weights='IMAGENET1K_V2')
        num_features = self.resnet.fc.in_features
        my_stuff = []
        for i in range(num_hidden):
            my_stuff.append(nn.Linear(num_features, num_features // 2))
            my_stuff.append(nn.ReLU())
            my_stuff.append(nn.Dropout(0.5))
            num_features = num_features // 2
        my_stuff.append(nn.Linear(num_features, num_classes))
        self.resnet.fc = nn.Sequential(*my_stuff)
        if freeze_pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        return x


class ExtendedResNet(nn.Module):
    def __init__(self, num_classes=100, freeze_pretrained=False):
        super(ExtendedResNet, self).__init__()
        self.resnet = models.resnet101(weights='IMAGENET1K_V2')
        if freeze_pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        self.extra_fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.extra_fc(x)
        return x


class Trainer:
    def __init__(self, batch_size=32, num_epochs=25, learning_rate=0.001,
                 in_weights=None, out_weights='./result/a.pth',
                 out_fig='./tmp.png'):
        self.root_dir = './data'
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.in_weights = in_weights
        self.out_weights = out_weights
        self.out_fig = out_fig
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ]),
            'test': transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.CenterCrop(384),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
            ])
        }

        self.image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.root_dir, x), self.data_transforms[x]
            ) for x in ['train', 'val']
        }
        self.image_loaders = {
            x: DataLoader(self.image_datasets[x],
                          batch_size=batch_size,
                          shuffle=True if x == 'train' else False,
                          num_workers=4)
            for x in ['train', 'val']
        }
        self.dataset_sizes = {
            x: len(self.image_datasets[x]) for x in ['train', 'val']
        }
        self.class_names = self.image_datasets['val'].classes

        self.model = self.initialize_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate, momentum=0.9, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def initialize_model(self):
        model = MyResNet(num_hidden=0, freeze_pretrained=False)
        # model = ExtendedResNet(
        #     num_classes=len(self.class_names), freeze_pretrained=False
        # )
        model = model.to(self.device)
        return model

    def train(self, training=True):
        if self.in_weights is not None:
            self.model.load_state_dict(
                torch.load(
                    self.in_weights,
                    map_location=self.device, weights_only=True
                )
            )
            logger.info(f'Load weights from {self.in_weights}!')
        if not training:
            return self.model

        best_val_acc = 0.0
        best_epoch = 0
        for epoch in range(self.num_epochs):
            self.model.train()

            running_loss = 0.0
            running_corrects = 0
            pbar = tqdm(self.image_loaders['train'], desc='Phase train')
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            self.scheduler.step()

            epoch_train_loss = running_loss / self.dataset_sizes['train']
            epoch_train_acc = \
                running_corrects.double() / self.dataset_sizes['train']

            self.train_losses.append(epoch_train_loss)
            self.train_accs.append(epoch_train_acc.item())
            self.model.eval()

            running_loss = 0.0
            running_corrects = 0
            pbar = tqdm(self.image_loaders['val'], desc='Phase val')
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_val_loss = running_loss / self.dataset_sizes['val']
            epoch_val_acc = \
                running_corrects.double() / self.dataset_sizes['val']
            self.val_losses.append(epoch_val_loss)
            self.val_accs.append(epoch_val_acc.item())

            logger.info(f'Epoch [{epoch+1}/{self.num_epochs}] \
                        Train Loss: {epoch_train_loss:.4f}, \
                        Train Acc: {epoch_train_acc:.4f}, \
                        Val Acc: {epoch_val_acc:.4f}')
            self.plot_learning_curves(epoch + 1)

            # Save the best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.out_weights)
                logger.info(f'New model saved to {self.out_weights}!')

        # Load best model weights
        logger.info(f'Model weights are epoch #{best_epoch + 1} \
                    with val acc {best_val_acc:.4f}.')
        self.model.load_state_dict(
            torch.load(
                self.out_weights,
                map_location=self.device,
                weights_only=True
            )
        )
        return self.model

    def test(self, output_file='./prediction.csv'):
        self.model.eval()

        self.image_datasets['test'] = TestDataset(
            os.path.join(self.root_dir, 'test'),
            transform=self.data_transforms['test']
        )
        self.image_loaders['test'] = DataLoader(
            self.image_datasets['test'],
            batch_size=32,
            shuffle=False,
            num_workers=4
        )

        results = []
        with torch.no_grad():
            pbar = tqdm(self.image_loaders['test'], desc='Evaluating')
            for inputs, names in pbar:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()
                results.extend(
                    zip(names, [self.class_names[p] for p in preds])
                )

        df = pd.DataFrame(results, columns=['image_name', 'pred_label'])
        df.to_csv(output_file, index=False)
        logger.info(f'Inference result is saved to {output_file}')

    def plot_learning_curves(self, epoch):
        epochs = range(1, epoch + 1)
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, 'b', label='Training Accuracy')
        plt.plot(epochs, self.val_accs, 'r', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.suptitle(f'{self.out_weights} {self.learning_rate}')
        plt.savefig(self.out_fig)
        plt.close()


def main(args):
    trainer = Trainer(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        in_weights=args.in_weights,
        out_weights=args.out_weights,
        out_fig=args.fig
    )
    logger.info(f'Device: {trainer.device}')
    trainer.train(training=args.train)
    trainer.test(output_file=args.csv)
    print("end of main")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--in_weights', type=str, default=None)
    parser.add_argument('--out_weights', type=str, default='./result/0.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--fig', type=str, default='./lc.png')
    parser.add_argument('--csv', type=str, default='./prediction.csv')
    args = parser.parse_args()
    main(args)
