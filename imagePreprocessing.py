import os
import cv2
import torch
import numpy as np
# import mediapipe as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# MediaPipe Configuration
mp_hands = np.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = np.solutions.drawing_utils


# Custom Dataset Class
class LandmarkDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._create_samples()
        self.augment = augment

    def _process_landmarks(self, image_path):
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([[lm.x * width, lm.y * height, lm.z]
                                  for lm in hand_landmarks.landmark]).flatten()
            if self.augment:
                noise = np.random.normal(0, 1, landmarks.shape) * 0.01
                landmarks += noise  # Add slight jitter
            return landmarks

        return np.zeros(63)  # Zero padding for no hand detected

    def _create_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img_file in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_file)
                samples.append((img_path, self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        landmarks = self._process_landmarks(img_path)

        # Load and preprocess image for VGG16
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0  # Normalize & reshape

        return image, torch.FloatTensor(landmarks), label


# Hybrid Model
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Pretrained VGG16 Model
        self.vgg16 = models.vgg16(pretrained=True)
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # Feedforward Layers
        self.fc1 = nn.Linear(128 + 63, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        landmarks, images = x
        image_features = self.vgg16.features(images)
        image_features = torch.flatten(image_features, start_dim=1)
        image_features = self.vgg16.classifier(image_features)

        x = torch.cat((image_features, landmarks), dim=1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.classifier(x)
        return x


# Training Configuration
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 36
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Dataset and DataLoader
train_dataset = LandmarkDataset('augmented_data/train', augment=True)
val_dataset = LandmarkDataset('augmented_data/val', augment=False)
test_dataset = LandmarkDataset('augmented_data/test', augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model, Optimizer, and Loss Function
model = HybridModel(NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

train_losses, val_accuracies = [], []

# Training and Evaluation
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, landmarks, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model((landmarks, images))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    scheduler.step()

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, landmarks, labels in val_loader:
            images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
            outputs = model((landmarks, images))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    val_accuracies.append(accuracy)
    print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Val Acc: {accuracy:.4f}")

# Save Model
# Ensure class mapping is saved correctly
torch.save({
    'model_state_dict': model.state_dict(),
    'class_mapping': train_dataset.class_to_idx,  # This is critical
    'num_classes': NUM_CLASSES  # Additional safety
}, 'translator-vgg16.h5')


# Save Performance Metrics
with open("performance_metrics.txt", "w") as f:
    f.write(f"Validation Accuracy: {val_accuracies}\n")

# Train vs Validation Accuracy Plot
plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
print("Training complete! Model and metrics saved.")
