"""
Fine-tune ResNet on labeled MRI images. Run once (or whenever you add data).

Folder layout (one sub-folder per diagnosis, any number of classes):
  dataset/
    glioma/       *.jpg
    meningioma/   *.jpg
    pituitary/    *.jpg
    notumor/      *.jpg

Run:  python train.py dataset
Output: model.pt  (weights + class names)
"""
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset"
EPOCHS, BATCH, LR = 8, 32, 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize((224, 224)),
    transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(), norm,
])
val_tf = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize((224, 224)),
    transforms.ToTensor(), norm,
])

full_train = datasets.ImageFolder(DATA, transform=train_tf)
full_val = datasets.ImageFolder(DATA, transform=val_tf)
classes = full_train.classes
print("Classes:", classes, "| images:", len(full_train))

idx = list(range(len(full_train)))
random.seed(42)
random.shuffle(idx)
cut = int(0.8 * len(idx))
train_dl = DataLoader(
    Subset(full_train, idx[:cut]), batch_size=BATCH, shuffle=True, num_workers=0)
val_dl = DataLoader(
    Subset(full_val, idx[cut:]), batch_size=BATCH, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
best = 0.0

for ep in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    acc = correct / total
    print(f"Epoch {ep + 1}/{EPOCHS}  validation accuracy: {acc:.3f}")
    if acc > best:
        best = acc
        torch.save({"state": model.state_dict(),
                   "classes": classes}, "model.pt")

print(f"Best validation accuracy: {best:.3f}  -> saved model.pt")
