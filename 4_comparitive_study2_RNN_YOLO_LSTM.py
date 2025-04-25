import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm

# Constants
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SOURCE_DIR = 'augmented_data'
TRAIN_DIR = os.path.join(SOURCE_DIR, 'train')
TEST_DIR = os.path.join(SOURCE_DIR, 'test')
VAL_DIR = os.path.join(SOURCE_DIR, 'val')

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

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# RNN model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))

        if self.fc1 is None or self.fc2 is None:
            flattened_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(flattened_size, 128)
            self.fc2 = nn.Linear(128, len(train_dataset.classes))

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# YOLO-like model
class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))

        if self.fc1 is None or self.fc2 is None:
            flattened_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(flattened_size, 128)
            self.fc2 = nn.Linear(128, len(train_dataset.classes))

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# LSTM model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # Convolution layers to extract features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self._get_lstm_input_size(), hidden_size=128, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, len(train_dataset.classes))

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))

        # Flatten the output to pass it to LSTM
        x = x.view(x.size(0), -1)  # Flatten (batch_size, channels*height*width)

        # Add extra dimension to use LSTM (batch_size, sequence_length, input_size)
        x = x.unsqueeze(1)  # Making it (batch_size, 1, channels*height*width)

        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)

        # Take the last hidden state output of LSTM
        x = hn[-1]

        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def _get_lstm_input_size(self):
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)  # Batch size = 1, 3 channels, 224x224 image
            x = self.pool(nn.functional.relu(self.conv1(sample_input)))
            x = self.pool2(nn.functional.relu(self.conv2(x)))
            return x.view(1, -1).shape[1]  # Get flattened size


# Train and evaluate models
results = {}
loaders = {'train': train_loader, 'test': test_loader}

rnn_model = RNN().to(device)
results['RNN'] = train_and_evaluate_model(rnn_model, 'RNN', device, loaders)

yolo_model = YOLO().to(device)
results['YOLO'] = train_and_evaluate_model(yolo_model, 'YOLO', device, loaders)

lstm_model = LSTM().to(device)
results['LSTM'] = train_and_evaluate_model(lstm_model, 'LSTM', device, loaders)

# Save the final results to a file
save_results_to_file(results)
# Print results
print("\nComparative Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy={metrics[0]:.4f}, Precision={metrics[1]:.4f}")
