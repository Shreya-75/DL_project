import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm

# Constants
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SOURCE_DIR = 'augmented_data'
TARGET_DIR = 'augmented_data_split'
TRAIN_DIR = os.path.join(TARGET_DIR, 'train')
TEST_DIR = os.path.join(TARGET_DIR, 'test')
VAL_DIR = os.path.join(TARGET_DIR, 'val')

# Function to create train, test, and validation datasets
def create_train_test_val_datasets(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        if os.path.isdir(folder_path):
            # Create subfolders for train, test, and validation
            train_folder = os.path.join(target_dir, 'train', folder)
            test_folder = os.path.join(target_dir, 'test', folder)
            val_folder = os.path.join(target_dir, 'val', folder)
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)
            os.makedirs(val_folder, exist_ok=True)

            # Get all image files in the folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'jpeg'))]
            total_images = len(image_files)

            # Shuffle and split the images
            np.random.shuffle(image_files)
            train_count = int(total_images * train_ratio)
            val_count = int(total_images * val_ratio)
            train_files = image_files[:train_count]
            val_files = image_files[train_count:train_count + val_count]
            test_files = image_files[train_count + val_count:]

            # Move files to train, test, and validation folders
            for file in train_files:
                shutil.copy(os.path.join(folder_path, file), os.path.join(train_folder, file))
            for file in val_files:
                shutil.copy(os.path.join(folder_path, file), os.path.join(val_folder, file))
            for file in test_files:
                shutil.copy(os.path.join(folder_path, file), os.path.join(test_folder, file))

# Create train, test, and validation datasets
create_train_test_val_datasets(SOURCE_DIR, TARGET_DIR)

# Data transforms
data_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transforms)
test_dataset = datasets.ImageFolder(TEST_DIR, data_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, data_transforms)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to calculate the output size of convolutional and pooling layers
def calculate_output_size():
    dummy_model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.MaxPool2d(2, 2)
    )
    dummy_input = torch.randn(1, 3, 224, 224)
    output = dummy_model(dummy_input)
    flattened_size = output.shape[1] * output.shape[2] * output.shape[3]
    return flattened_size

# Calculate the flattened size
flattened_size = calculate_output_size()
print(f"Flattened size: {flattened_size}")

# Updated train_and_evaluate_model function with tqdm
def train_and_evaluate_model(model, model_name, device, loaders):
    print(f"Training {model_name}...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, data in enumerate(progress_bar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        print(f"Epoch {epoch + 1}, Loss: {running_loss / (i + 1):.4f}")

    # Evaluate the model
    model.eval()
    correct = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data in loaders['test']:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = correct / len(test_loader.dataset)
    precision = precision_score(true_labels, predicted_labels, average='weighted')

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    return accuracy, precision

# Save results to a file
def save_results_to_file(results, file_name="results.txt"):
    with open(file_name, "w") as file:
        file.write("Model Comparison Results\n")
        file.write("========================\n")
        for model_name, metrics in results.items():
            file.write(f"{model_name}: Accuracy={metrics[0]:.4f}, Precision={metrics[1]:.4f}\n")



# Models to evaluate
results = {}

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, len(train_dataset.classes))

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, flattened_size)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Transfer Learning (VGG16)
vgg_model = vgg16(pretrained=True)
for param in vgg_model.parameters():
    param.requires_grad = False
num_ftrs = vgg_model.classifier[6].in_features  # Access the final layer's input features
vgg_model.classifier[6] = nn.Linear(num_ftrs, len(train_dataset.classes))


# Train and evaluate models
results = {}

cnn_model = CNN().to(device)
loaders = {'train': train_loader, 'test': test_loader}
results['CNN'] = train_and_evaluate_model(cnn_model, 'CNN', device, loaders)

transfer_model = vgg_model.to(device)
results['VGG16'] = train_and_evaluate_model(transfer_model, 'VGG16', device, loaders)

# Save the final results to a file
save_results_to_file(results)

# Print results
print("\nComparative Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy={metrics[0]:.4f}, Precision={metrics[1]:.4f}")