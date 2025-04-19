import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import argparse
from model import Cnet, get_activation
import gc
# from tqdm import tqdm

def train_and_eval(args, logging=False):

    print("Memory used before starting : ", torch.cuda.memory_allocated()/1000000, torch.cuda.memory_reserved()/1000000)

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip() if args.data_aug else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(10) if args.data_aug else transforms.Lambda(lambda x: x),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

    train_data = torchvision.datasets.ImageFolder(root='/speech/shoutrik/Databases/inaturalist_12K/train', transform=train_transform)
    val_data = torchvision.datasets.ImageFolder(root='/speech/shoutrik/Databases/inaturalist_12K/valid', transform=test_transform)
    test_data = torchvision.datasets.ImageFolder(root='/speech/shoutrik/Databases/inaturalist_12K/test', transform=test_transform)

    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conv_activation = get_activation(args.conv_activation)
    dense_activation = get_activation(args.dense_activation)

    model = Cnet(
        # in_dims=args.in_dims,
        conv_layer_config=list(zip(args.n_filter, args.filter_size)),
        # filter_org=args.filter_org,
        batch_norm=args.batch_norm,
        dropout_prob=args.dropout_prob,
        dense_unit=args.dense_size,
        activation=conv_activation,
        # dense_activation=dense_activation,
        num_classes=len(train_data.classes)
    ).to(device)

    print(model)

    param_count = sum(p.numel() for p in model.parameters())
    print("Number of parameters:", param_count)
    print("Memory used : ", torch.cuda.memory_allocated()/1000000, torch.cuda.memory_reserved()/1000000)

    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0

        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        for images, labels in trainloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            out_ = model(images)
            loss = criterion(out_, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1} peak memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{args.n_epochs} | Train Loss: {train_loss/len(trainloader):.4f} | Val Loss: {val_loss/len(valloader):.4f} | Val Acc: {val_acc:.2f}%")

        if logging:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": float(train_loss / len(trainloader)),
                "val_loss": float(val_loss / len(valloader)),
                "val_accuracy": float(val_acc)
            })

        # Cleanup after epoch
        gc.collect()
        torch.cuda.empty_cache()

    # Testing
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss/len(testloader):.4f} | Test Acc: {test_acc:.2f}%")

    if logging:
        wandb.log({
            "test_loss": float(test_loss / len(testloader)),
            "test_accuracy": float(test_acc)
        })

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dims", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--conv_activation", type=str, default="GELU")
    parser.add_argument("--dense_activation", type=str, default="ReLU")
    parser.add_argument("--dense_size", type=int, default=1024)
    parser.add_argument("--filter_size", nargs='+', type=int, default=[7, 5, 5, 3, 3])
    parser.add_argument("--n_filter", nargs='+', type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--filter_org", type=str, default="same")
    parser.add_argument("--batch_norm", type=bool, default=True)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--data_aug", type=bool, default=True)
    args = parser.parse_args()

    train_and_eval(args, logging=False)