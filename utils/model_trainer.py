import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ImprovedEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ImprovedEmotionCNN, self).__init__()
        # 更深的卷积网络结构
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # 更复杂的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.image_paths = []
        self.labels = []
        
        for label, emotion in enumerate(self.classes):
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(emotion_dir, img_name))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 更复杂的数据增强
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def train_model(self, data_dir, model_save_path, epochs=100, batch_size=64):
        train_dataset = EmotionDataset(data_dir, self.transform)
        if len(train_dataset) == 0:
            raise ValueError("没有找到训练图像，请检查数据集路径和结构")
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = ImprovedEmotionCNN().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            # 使用验证损失调整学习率
            scheduler.step(epoch_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            
            # 保存最佳模型
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), model_save_path)
                print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        print(f'Training completed. Best accuracy: {best_acc:.2f}%')
        return model