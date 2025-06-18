#这是对模型的测试文件，单独运行
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np

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

class ModelTester:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        # 加载模型
        self.model = ImprovedEmotionCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def test_model(self, test_data_dir):
        test_dataset = EmotionDataset(test_data_dir, self.transform)
        if len(test_dataset) == 0:
            raise ValueError("没有找到测试图像，请检查数据集路径和结构")
            
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        correct = 0
        total = 0
        class_correct = [0] * 7
        class_total = [0] * 7
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 计算每个类别的准确率
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        overall_accuracy = 100 * correct / total
        print(f'Overall Test Accuracy: {overall_accuracy:.2f}%')
        print('\nPer-class accuracy:')
        for i in range(7):
            if class_total[i] > 0:
                print(f'{test_dataset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')
            else:
                print(f'{test_dataset.classes[i]}: No samples found')
        
        return overall_accuracy

# 使用示例
if __name__ == "__main__":
    class ImprovedEmotionCNN(nn.Module):
        def __init__(self, num_classes=7):
            super(ImprovedEmotionCNN, self).__init__()
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
    model_path = "models\\custom_model2.pth"  # 替换为需要测试的模型路径（结构需要和上方相同）
    test_data_dir = "test"  # 替换真实测试集路径
    
    tester = ModelTester(model_path)
    accuracy = tester.test_model(test_data_dir)