import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Step 1: Load and prepare data ---
X = np.load("sequence_data.npy")
y = np.load("targets.npy")


# --- Step 2: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 3: Device selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Step 4: Convert to tensors, move to device ---
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)

# --- Step 5: Wrap training data for batching ---
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# --- Step 6: Define the network ---
class ResourceTrendPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=256, num_layers=3, batch_first=True)
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = ResourceTrendPredictor().to(device)

# --- Step 7: Loss function and optimizer ---
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Step 8: Training loop, with batching ---
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

# --- Step 9: Evaluate on test set ---
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    predicted_classes = (test_predictions >= 0.5).float()
    accuracy = (predicted_classes == y_test).float().mean()
    test_loss = criterion(test_predictions, y_test)
    mae = torch.mean(torch.abs(test_predictions - y_test))
    print(f"Test Loss: {test_loss.item():.4f}, MAE: {mae.item():.4f}, Accuracy: {accuracy.item() *100:.2f}%")

# --- Step 10: Save the trained model ---
torch.save(model.state_dict(), "resource_model.pth")
print("Model saved to resource_model.pth")