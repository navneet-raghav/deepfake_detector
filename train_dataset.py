import os
import sys
import argparse
import csv
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

try:
    import timm
except ImportError:
    print("[ERROR] timm not installed. Run:  pip install timm")
    sys.exit(1)


#Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("dataset_root")
parser.add_argument("classes_file")
parser.add_argument("result_root")
parser.add_argument("--model",      default="xception",
                    choices=["xception", "efficientnet_b0"],
                    help="Model to train (default: xception)")
parser.add_argument("--epochs",     type=int,   default=15)
parser.add_argument("--batch_size", type=int,   default=32)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--img_size",   type=int,   default=299)
parser.add_argument("--workers",    type=int,   default=2)

class JPEGCompression:
    """Simulate JPEG compression at random quality to prevent codec overfitting."""
    def __init__(self, quality_low=40, quality_high=95):
        self.low  = quality_low
        self.high = quality_high

    def __call__(self, img):
        import random
        q   = random.randint(self.low, self.high)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).copy()


class GaussianNoise:
    """Add camera-like sensor noise."""
    def __init__(self, std_max=12):
        self.std_max = std_max

    def __call__(self, img):
        import random
        std = random.uniform(0, self.std_max)
        if std < 1.0:
            return img
        arr   = np.array(img).astype(np.float32)
        noise = np.random.normal(0, std, arr.shape)
        arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class RandomBlur:
    """Simulate motion blur or out-of-focus shots."""
    def __init__(self, p=0.3, sigma_max=1.5):
        self.p         = p
        self.sigma_max = sigma_max

    def __call__(self, img):
        import random
        if random.random() > self.p:
            return img
        sigma = random.uniform(0.1, self.sigma_max)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        return img, label



def run_epoch(loader, model, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct    = 0
    total_n    = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            out  = model(imgs)
            loss = criterion(out, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total_n    += imgs.size(0)

    return total_loss / total_n, correct / total_n



def main():
    args = parser.parse_args()

    os.makedirs(args.result_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Model  : {args.model}")

    with open(args.classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    num_classes = len(classes)
    print(f"[INFO] Classes: {classes}")

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]


    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        RandomBlur(p=0.3),
        GaussianNoise(std_max=12),
        JPEGCompression(quality_low=40, quality_high=95),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


    #Dataset
    full_dataset = datasets.ImageFolder(args.dataset_root)
    print(f"[INFO] Class mapping : {full_dataset.class_to_idx}")

    total      = len(full_dataset)
    val_size   = int(total * 0.2)
    train_size = total - val_size

    torch.manual_seed(42)
    train_idx, val_idx = torch.utils.data.random_split(
        range(total), [train_size, val_size]
    )
    train_idx = list(train_idx)
    val_idx   = list(val_idx)


    full_dataset.transform        = None
    full_dataset.target_transform = None

    train_ds = SubsetWithTransform(full_dataset, train_idx, train_tf)
    val_ds   = SubsetWithTransform(full_dataset, val_idx,   val_tf)

    print(f"[INFO] Train samples : {len(train_ds)}")
    print(f"[INFO] Val samples   : {len(val_ds)}")

    train_labels  = [full_dataset.targets[i] for i in train_idx]
    class_counts  = np.bincount(train_labels)
    print(f"[INFO] Train counts  : {dict(zip(classes, class_counts))}")

    class_weights  = 1.0 / (class_counts.astype(float) + 1e-8)
    sample_weights = np.array([class_weights[lbl] for lbl in train_labels])
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))

    #Model
    print(f"[INFO] Loading {args.model} with ImageNet weights ...")
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    print(f"[INFO] Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    #Loss 
    loss_w    = torch.tensor(class_weights / class_weights.sum() * num_classes,
                             dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_w)



    history          = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc     = 0.0
    best_epoch       = 0
    best_model_path  = os.path.join(args.result_root, f"best_model_{args.model}.pth")
    final_model_path = os.path.join(args.result_root, f"final_model_{args.model}.pth")

    csv_path   = os.path.join(args.result_root, f"training_log_{args.model}.csv")
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc"])


    def log_epoch(epoch, phase, tl, ta, vl, va):
        nonlocal best_val_acc, best_epoch
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        csv_writer.writerow([epoch+1, phase, f"{tl:.4f}", f"{ta:.4f}", f"{vl:.4f}", f"{va:.4f}"])
        csv_file.flush()

        mark = ""
        if va > best_val_acc:
            best_val_acc = va
            best_epoch   = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            mark = "  <- best"

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}{mark}")


    # Phase 1: Head only 
    phase1_end = min(10, args.epochs)
    print(f"\n[PHASE 1] Head only — base frozen  (epochs 1–{phase1_end})")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True

    opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, "min", factor=0.5, patience=3)

    for epoch in range(phase1_end):
        tl, ta = run_epoch(train_loader, model, criterion, device, opt1)
        vl, va = run_epoch(val_loader,   model, criterion, device, None)
        sch1.step(vl)
        log_epoch(epoch, 1, tl, ta, vl, va)

    #Phase 2: Full fine tune
    if args.epochs > 10:
        print(f"\n[PHASE 2] Full fine-tune — all layers  (epochs 11–{args.epochs})")

        for p in model.parameters():
            p.requires_grad = True

        opt2 = optim.Adam(model.parameters(), lr=args.lr / 10)
        sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, "min", factor=0.5, patience=3)

        for epoch in range(10, args.epochs):
            tl, ta = run_epoch(train_loader, model, criterion, device, opt2)
            vl, va = run_epoch(val_loader,   model, criterion, device, None)
            sch2.step(vl)
            log_epoch(epoch, 2, tl, ta, vl, va)

    csv_file.close()

    torch.save(model.state_dict(), final_model_path)
    print(f"\n[INFO] Best model  -> {best_model_path}  (epoch {best_epoch}, {best_val_acc*100:.2f}%)")
    print(f"[INFO] Final model -> {final_model_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{args.model} — training curves", fontsize=13)
    ep_x = range(1, len(history["train_acc"]) + 1)

    for ax, key_tr, key_va, title, ylabel in [
        (axes[0], "train_acc",  "val_acc",  "Accuracy", "Accuracy"),
        (axes[1], "train_loss", "val_loss", "Loss",     "Loss"),
    ]:
        ax.plot(ep_x, history[key_tr], label="Train", color="royalblue")
        ax.plot(ep_x, history[key_va], label="Val",   color="coral")
        if args.epochs > 10:
            ax.axvline(x=10, color="gray", linestyle="--", linewidth=1, label="Phase 2")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    if "acc" in "train_acc":
        axes[0].set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(args.result_root, f"training_curves_{args.model}.png")
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"[INFO] Curves      -> {plot_path}")

    print(f"\n{'='*52}")
    print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"{'='*52}")
    print("Training complete!")


if __name__ == '__main__':
    main()