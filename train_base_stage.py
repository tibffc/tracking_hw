import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import json
import os
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, fix_paths=False):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        if fix_paths:
            print(f" Fixing paths in {csv_file}...")
            self.data['file_name'] = self.data['file_name'].str.replace('train_data/', 'test_data/', regex=False)
            print(f"   Fixed first file: {self.data.iloc[0]['file_name']}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.root_dir / self.data.iloc[idx]['file_name']
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Трансформации
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка данных
train_dataset = ImageDataset(
    csv_file='ai-vs-human-generated-dataset-hw/Train_1/train.csv',
    root_dir='ai-vs-human-generated-dataset-hw/Train_1',
    transform=train_transform
)

test_dataset = ImageDataset(
    csv_file='ai-vs-human-generated-dataset-hw/Test_1/test.csv',
    root_dir='ai-vs-human-generated-dataset-hw/Test_1',
    transform=test_transform,
    fix_paths=True
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print(f'Train dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

# Модель
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# Обучение
num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_losses = []
train_accs = []
train_f1s = []
train_precisions = []
train_recalls = []
val_losses = []
val_accs = []
val_f1s = []
val_precisions = []
val_recalls = []

print("Starting training...")
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)
    
    # ========== ОБУЧЕНИЕ ==========
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc='Training')):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = accuracy_score(all_labels, all_preds)
    epoch_train_f1 = f1_score(all_labels, all_preds, average='weighted')
    epoch_train_precision = precision_score(all_labels, all_preds, average='weighted')
    epoch_train_recall = recall_score(all_labels, all_preds, average='weighted')
    
    # ========== ВАЛИДАЦИЯ ==========
    model.eval()
    running_val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
    
    epoch_val_loss = running_val_loss / len(test_loader.dataset)
    epoch_val_acc = accuracy_score(all_val_labels, all_val_preds)
    epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
    epoch_val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
    epoch_val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
    
    scheduler.step()
    
    # Сохраняем метрики
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)
    train_f1s.append(epoch_train_f1)
    train_precisions.append(epoch_train_precision)
    train_recalls.append(epoch_train_recall)
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)
    val_f1s.append(epoch_val_f1)
    val_precisions.append(epoch_val_precision)
    val_recalls.append(epoch_val_recall)
    
    print(f'Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f}')
    print(f'Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}, Precision: {epoch_val_precision:.4f}, Recall: {epoch_val_recall:.4f}')

print('\nTraining completed!')

# Сохраняем модель
os.makedirs('models', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'version': '1.0',
    'timestamp': datetime.datetime.now().isoformat()
}, 'models/resnet18_ai_detector_v1.0.pth')


final_metrics = {
    'train_loss': train_losses,
    'train_accuracy': train_accs,
    'train_f1': train_f1s,
    'train_precision': train_precisions,
    'train_recall': train_recalls,
    'val_loss': val_losses,
    'val_accuracy': val_accs,
    'val_f1': val_f1s,
    'val_precision': val_precisions,
    'val_recall': val_recalls,
    'best_val_accuracy': max(val_accs),
    'best_val_f1': max(val_f1s),
    'final_val_accuracy': val_accs[-1],
    'final_val_f1': val_f1s[-1],
    'final_val_precision': val_precisions[-1],
    'final_val_recall': val_recalls[-1]
}

with open('models/metrics_v1.0.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)

print(" Base model training completed with all metrics!")