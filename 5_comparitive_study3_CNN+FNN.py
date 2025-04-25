import os
import cv2
import torch
import numpy as np
# import mediapipe as mp
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision import transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe Hands
mp_hands = np.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Log file setup
log_file = open("training_logs.txt", "w")


# Function to extract landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    landmarks = np.zeros((21, 3))

    if results.multi_hand_landmarks:
        for i, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]

    return landmarks.flatten()  # Shape (63,)


# Dataset Class
class HandGestureDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))
        self.file_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            for file in os.listdir(class_path):
                self.file_paths.append(os.path.join(class_path, file))
                self.labels.append(idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)

        # Extract landmarks
        landmarks = extract_landmarks(image)

        # Apply transformations
        image = transform(image)

        return image, torch.tensor(landmarks, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# CNN + Feedforward Model
class GestureRecognitionModel(nn.Module):
    def __init__(self, num_classes=36):
        super(GestureRecognitionModel, self).__init__()

        # CNN Layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten_size = 128 * (128 // 8) * (128 // 8)  # Compute size after CNN layers

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size + 63, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)  # Reduced dropout

    def forward(self, image, landmarks):
        cnn_features = self.cnn(image)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)

        features = torch.cat((cnn_features, landmarks), dim=-1)

        out = self.fc1(features)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)

        return out


# Training function
def train_model(model, train_loader, val_loader, num_epochs=15):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing added
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)  # SGD with momentum

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [],[]
    best_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, landmarks, labels in tqdm_bar:
            images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            tqdm_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, landmarks, labels in val_loader:
                images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
                outputs = model(images, landmarks)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        log_msg = f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}\n"
        print(log_msg)
        log_file.write(log_msg)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "gesture_recognition_model.pth")

    log_file.close()

    # Save confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(set(all_labels))), yticklabels=range(len(set(all_labels))))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Plot Loss Graphs
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy Graphs
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("accuracy_loss_plot.png")
    plt.show()


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, landmarks, labels in test_loader:
            images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
            outputs = model(images, landmarks)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")


# Load dataset
train_dataset = HandGestureDataset('augmented_data_split/train')
val_dataset = HandGestureDataset('augmented_data_split/val')
test_dataset = HandGestureDataset('augmented_data_split/test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
model = GestureRecognitionModel().to(device)

# Train model
train_model(model, train_loader, val_loader)

# Evaluate model
evaluate_model(model, test_loader)
