import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import copy

# 1. Configuration
DATA_DIR = r"c:\Users\lenovo\Downloads\comms proj\comms proj"
CHUNK_SIZE = 1024  # Split the 4096 length samples into chunks of 1024
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0005

CLASSES = [
    'dsbtc', 'dsbsc', 'ssbsc', 'fm',  # Analog
    'ask', 'fsk', 'bpsk', 'qpsk', '8psk', '16qam', '64qam', 'msk' # Digital
]
NUM_CLASSES = len(CLASSES)

def extract_label_from_filename(filename):
    # Remove digits to match with CLASSES, handle possible extensions
    clean_name = "".join([c for c in filename.split('.')[0] if not c.isdigit() and c != '-'])
    for c in CLASSES:
        if c == clean_name.lower() or c in filename.lower():
            return c
    return None

def extract_snr_from_filename(filename):
    match = re.search(r'(-?\d+)dB', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

# Add AWGN to signal to simulate SNR if SNR isn't present
def awgn(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    power = np.mean(np.abs(signal) ** 2, axis=-1, keepdims=True)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

print("Loading data...")
all_data = []
all_labels = []
all_snrs = []
files = os.listdir(DATA_DIR)

# Filter valid files
valid_files = [f for f in files if os.path.isfile(os.path.join(DATA_DIR, f)) and extract_label_from_filename(f)]
print(f"Found {len(valid_files)} valid data files.")

for f in valid_files:
    label = extract_label_from_filename(f)
    snr = extract_snr_from_filename(f)
    
    filepath = os.path.join(DATA_DIR, f)
    try:
        # Load Raw IQ values, typically np.complex64
        data = np.fromfile(filepath, dtype=np.complex64)
        
        # Split into chunks of CHUNK_SIZE
        num_chunks = len(data) // CHUNK_SIZE
        for i in range(num_chunks):
            chunk = data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            all_data.append(chunk)
            all_labels.append(CLASSES.index(label))
            all_snrs.append(snr)
            
    except Exception as e:
        print(f"Error loading {f}: {e}")

if not all_data:
    print("No data loaded. Please check dataset path and format.")
    exit()

X = np.array(all_data)
y = np.array(all_labels)
snrs = np.array(all_snrs)

has_snr = any(s is not None for s in snrs)

# Formatting X for CNN: Shape (Batch, Channels, Length) -> (N, 2, CHUNK_SIZE)
X_concat = np.stack([np.real(X), np.imag(X)], axis=1)

print(f"Dataset Shape: {X_concat.shape}")

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_concat, y, test_size=0.2, random_state=42, stratify=y)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# 2. Define PyTorch Model (1D CNN architecture similar to VT-CNN)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class IQClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super(IQClassifier, self).__init__()
        # Novel 1D-ResNet Architecture designed for SDR IQ mapping
        self.entry = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.res1 = ResidualBlock(64)
        self.down1 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.MaxPool1d(2), nn.Dropout(0.2))
        self.res2 = ResidualBlock(128)
        self.down2 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.MaxPool1d(2), nn.Dropout(0.2))
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * (CHUNK_SIZE // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        amp = torch.sqrt(x[:, 0:1, :]**2 + x[:, 1:2, :]**2 + 1e-8)
        phase = torch.atan2(x[:, 1:2, :], x[:, 0:1, :])
        x_4ch = torch.cat([x, amp, phase], dim=1)
        
        x = self.entry(x_4ch)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")
model = IQClassifier(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 3. Training Loop
print("Starting training...")
train_losses = []
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    
    # Save best
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc*100:.2f}%")

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "iq_classifier_model.pth")
print("Model trained and saved as iq_classifier_model.pth")

# Plot Training Loss
plt.figure(figsize=(8,6))
plt.plot(train_losses, label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_loss.png")

# 4. Evaluation (Confusion Matrix)
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")

print(classification_report(all_targets, all_preds, target_names=CLASSES, zero_division=0))

# 5. Accuracy vs SNR Plot
snr_range = range(-10, 22, 2)
acc_vs_snr = []

print("Simulating Accuracy vs SNR (-10 dB to 20 dB)...")
model.eval()
for snr_val in snr_range:
    correct = 0
    total = 0
    
    # We will simulate noise on the raw test data
    # X_test has shape (N, 2, CHUNK_SIZE)
    snr_linear = 10 ** (snr_val / 10.0)
    power = np.mean(np.sum(X_test**2, axis=1, keepdims=True), axis=-1, keepdims=True)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power / 2) * np.random.randn(*X_test.shape)
    
    test_noisy_format = X_test + noise
    
    test_noisy_tensor = torch.tensor(test_noisy_format, dtype=torch.float32)
    eval_loader = DataLoader(TensorDataset(test_noisy_tensor, y_test_t), batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = correct / total
    acc_vs_snr.append(acc)
    print(f"SNR: {snr_val} dB, Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(8,6))
plt.plot(snr_range, acc_vs_snr, marker='o', linestyle='-')
plt.title("Classification Accuracy vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.ylim([0, 1.05])
plt.savefig("accuracy_vs_snr.png")
print("Accuracy vs SNR plot saved as accuracy_vs_snr.png")
print("All tasks completed successfully!")
