import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
import os
import gzip
import pickle
import requests
from io import BytesIO
import time
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, subdir, transform=None):
        self.root_dir = root_dir
        self.subdir = subdir
        self.transform = transform
        self.dir_path = os.path.join(root_dir, subdir)
        print(f"Initializing CustomDataset with {self.dir_path}")
        self.images = self._load_images()

    def _load_images(self):
        images = []
        subdirs = ['0', '1']
        for s in subdirs:
            subdir_path = os.path.join(self.dir_path, s) 
            if os.path.isdir(subdir_path):
                print(f"Directory found: {subdir_path}")
                for img_name in os.listdir(subdir_path):
                    img_path = os.path.join(subdir_path, img_name)
                    if img_path.endswith('.png'):
                        label = int(s)
                        images.append((img_path, label))
                        print(f"Image found: {img_path}")
                    else:
                        print(f"File skipped (not .png): {img_path}")
            else:
                print(f"Subdirectory not found: {subdir_path}")

        print(f"Loaded {len(images)} images in total")
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((400, 400)),  # Resize image
    transforms.RandomCrop((224, 224)),  # Randomly crop image
    transforms.RandomRotation(3),  # Randomly rotate image
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Load datasets
train_set = CustomDataset(root_dir="/path/Data", subdir="train", transform=transform)
test_set = CustomDataset(root_dir="/path/Data", subdir="test", transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

client_id = "client_2"

# models
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# functions
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def save_model_to_gzip(model, filepath):
    parameters = model.state_dict()
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(parameters, f)

def load_model_from_gzip(filepath):
    with gzip.open(filepath, 'rb') as f:
        parameters = pickle.load(f)
    return parameters

def restore_model_from_gzip(model, filepath):
    parameters = load_model_from_gzip(filepath)
    model.load_state_dict(parameters)
    return model

def get_avg_model_from_server(filepath,epoch):
    url = 'http://127.0.0.1:5000/model?epoch='+str(epoch)
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False

def retry_get_avg_model_from_server(t, max_retries=10, retry_interval=30):
    for attempt in range(max_retries):
        if get_avg_model_from_server(f"{t}.pkl.gz",t):
            return True
        time.sleep(retry_interval)
    return False

def upload_parameters_to_server(model):
    global client_id
    url = f"http://127.0.0.1:5000/upload?client_id={client_id}"
    memfile = BytesIO()
    with gzip.GzipFile(fileobj=memfile, mode='wb') as f:
        pickle.dump(model.state_dict(), f)
    memfile.seek(0)
    files = {'model': memfile}
    response = requests.post(url, files=files)
    return response.status_code == 200

model = SimpleNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    
    # 上傳模型參數到server
    if upload_parameters_to_server(model):
        print("Parameters uploaded successfully.")
    else:
        print("Failed to upload parameters.")

    # 嘗試從server獲取平均後的模型
    if retry_get_avg_model_from_server(t):
        restore_model_from_gzip(model, f"{t}.pkl.gz")
    else:
        print("fail")
    test(test_loader, model, loss_fn)
    
print("Done!")