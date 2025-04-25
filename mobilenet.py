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

# CPU Optimization
torch.set_num_threads(os.cpu_count() or 4)
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MediaPipe Configuration
mp_hands = np.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# Optimized Dataset Class with 224x224 support
class LandmarkDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._create_samples()
        self.augment = augment

    def _process_landmarks(self, image_path):
        image = cv2.imread(image_path)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            h, w = image.shape[:2]
            landmarks = np.array([[lm.x * w, lm.y * h, lm.z]
                                  for lm in results.multi_hand_landmarks[0].landmark]).flatten()
            if self.augment:
                landmarks += np.random.normal(0, 0.01, landmarks.shape)
            return landmarks.astype(np.float32)  # Ensure float32
        return np.zeros(63, dtype=np.float32)

    def _create_samples(self):
        return [(os.path.join(root, f), self.class_to_idx[cls])
                for cls in self.classes
                for root, _, files in os.walk(os.path.join(self.root_dir, cls))
                for f in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        landmarks = self._process_landmarks(img_path)

        # Load 224x224 image
        image = cv2.resize(cv2.imread(img_path), (224, 224))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, torch.tensor(landmarks, dtype=torch.float32), label


# MobileNetV2 Model for 224x224 input
class OptimizedASLModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze initial layers
        for param in self.mobilenet.features[:14].parameters():
            param.requires_grad = False

        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256)
        )

        # Landmark processing
        self.landmark_fc = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Final classifier
        self.final_fc = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        landmarks, images = x
        img_features = self.mobilenet(images)
        lm_features = self.landmark_fc(landmarks)
        combined = torch.cat((img_features, lm_features), dim=1)
        return self.final_fc(combined)


def main():
    # Training Configuration
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-5
    NUM_CLASSES = 36
    device = torch.device('cpu')

    # Initialize Data Loaders
    train_dataset = LandmarkDataset('augmented_data/train', augment=True)
    val_dataset = LandmarkDataset('augmented_data/val')
    test_dataset = LandmarkDataset('augmented_data/test')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True
    )

    # Model setup
    model = OptimizedASLModel(NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Training metrics
    train_losses = []
    val_accuracies, val_precisions, val_recalls = [], [], []
    best_val_acc = 0
    patience = 3
    no_improvement = 0

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for images, landmarks, labels in tqdm(train_loader,
                                              desc=f"Epoch {epoch + 1}/{EPOCHS}",
                                              mininterval=5):
            # Convert all to float32 explicitly
            images = images.to(device, non_blocking=True).float()
            landmarks = landmarks.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad()
            outputs = model((landmarks, images))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, landmarks, labels in val_loader:
                images = images.to(device).float()
                landmarks = landmarks.to(device).float()
                labels = labels.to(device).long()

                outputs = model((landmarks, images))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        val_accuracies.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)

        print(f"Epoch {epoch + 1} | "
              f"Train Loss: {train_losses[-1]:.4f} | "
              f"Val Acc: {accuracy:.4f} | "
              f"Precision: {precision:.4f} | "
              f"Recall: {recall:.4f}")

        # Early stopping
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            no_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_mapping': train_dataset.class_to_idx,
                'num_classes': NUM_CLASSES
            }, 'best_model.pth')
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Final test evaluation
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    model.eval()

    all_test_preds, all_test_labels = [], []
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

    with torch.no_grad():
        for images, landmarks, labels in test_loader:
            images = images.to(device).float()
            landmarks = landmarks.to(device).float()
            labels = labels.to(device).long()

            outputs = model((landmarks, images))
            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # Save results
    test_accuracy = accuracy_score(all_test_labels, all_test_preds)
    test_precision = precision_score(all_test_labels, all_test_preds, average='weighted')
    test_recall = recall_score(all_test_labels, all_test_preds, average='weighted')
    test_cm = confusion_matrix(all_test_labels, all_test_preds)

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_mapping': train_dataset.class_to_idx,
        'num_classes': NUM_CLASSES,
        'test_metrics': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall
        }
    }, 'asl_model.pth')

    with open("performance_metrics.txt", "w") as f:
        f.write(f"Validation Accuracies: {val_accuracies}\n")
        f.write(f"Validation Precisions: {val_precisions}\n")
        f.write(f"Validation Recalls: {val_recalls}\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")

    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Progress')
    plt.savefig('training_accuracy.png')
    plt.close()

    plt.figure(figsize=(15, 12))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    print("Training complete! Saved:")
    print("- asl_model.pth")
    print("- performance_metrics.txt")
    print("- training_accuracy.png")
    print("- confusion_matrix.png")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()
