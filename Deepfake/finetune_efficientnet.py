"""
Fine-tune EfficientNet-B0 on the existing real/fake face dataset.
Run:  python3 finetune_efficientnet.py
Saves:  model.pth  (replaces the old SimpleCNN weights)
"""

import os, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import DeepfakeDetector

# ─── Config ───────────────────────────────────────────────────────────────────
REAL_DIR   = "dataset/real"
FAKE_DIR   = "dataset/fake"
SAVE_PATH  = "model.pth"
EPOCHS     = 8
BATCH      = 16
LR         = 2e-4
VAL_SPLIT  = 0.15
SEED       = 42
IMG_SIZE   = 224
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
random.seed(SEED)

# ─── Augmentation ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─── Dataset ──────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples   # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_samples(directory, label):
    samples = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(root, f), label))
    return samples


all_samples = collect_samples(REAL_DIR, 0) + collect_samples(FAKE_DIR, 1)
random.shuffle(all_samples)

val_size   = int(len(all_samples) * VAL_SPLIT)
train_size = len(all_samples) - val_size
train_samples = all_samples[:train_size]
val_samples   = all_samples[train_size:]

train_ds = FaceDataset(train_samples, transform=train_transform)
val_ds   = FaceDataset(val_samples,   transform=val_transform)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

print(f"Training on {len(train_ds)} samples | Validating on {len(val_ds)} samples")

# ─── Model ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"Using device: {device}")

model = DeepfakeDetector(num_classes=2).to(device)

# Freeze backbone — only train the classifier head for the first 2 epochs
for name, param in model.backbone.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):

    # Unfreeze all layers after epoch 2 for full fine-tuning
    if epoch == 3:
        print("  → Unfreezing full backbone for fine-tuning")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=LR / 5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ── Train ──
    model.train()
    running_loss = correct = total = 0
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    train_acc  = 100 * correct / total
    train_loss = running_loss / len(train_dl)

    # ── Validate ──
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"  Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Saved best model (val_acc={val_acc:.1f}%)")

    scheduler.step()

print(f"\n🎉 Training complete! Best val accuracy: {best_val_acc:.1f}%")
print(f"   Model saved to: {SAVE_PATH}")
