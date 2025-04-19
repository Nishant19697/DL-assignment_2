import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import gc
from torchvision.models import vit_b_16, ViT_B_16_Weights

def train_and_eval(config=None):
    with wandb.init() as run:
        print("Memory used before starting : ", torch.cuda.memory_allocated()/1e6, torch.cuda.memory_reserved()/1e6)

        # using wandb.config to access hyperparameters
        batch_size = wandb.config.batch_size
        learning_rate = wandb.config.learning_rate
        weight_decay = wandb.config.weight_decay

        # loading the pre-trained ViT weights for finetuning
        weights = ViT_B_16_Weights.DEFAULT
        transform = weights.transforms()

        augmented_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_data = torchvision.datasets.ImageFolder(root='/speech/shoutrik/Databases/inaturalist_12K/train', transform=augmented_transform)
        val_data = torchvision.datasets.ImageFolder(root='/speech/shoutrik/Databases/inaturalist_12K/valid', transform=transform)
        test_data = torchvision.datasets.ImageFolder(root='/speech/shoutrik/Databases/inaturalist_12K/test', transform=transform)

        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the ViT model with the pre-trained weights
        model = vit_b_16(weights=weights)
        for param in model.parameters():
            param.requires_grad = False

        # here we are replacing the classifier head by inaturalist no. of classes
        model.heads.head = nn.Sequential(
            nn.Linear(model.heads.head.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, len(train_data.classes))
        )
        for param in model.heads.head.parameters():
            param.requires_grad = True

        model.to(device)

        print("Number of parameters:", sum(p.numel() for p in model.parameters()))
        print("Memory used : ", torch.cuda.memory_allocated()/1e6, torch.cuda.memory_reserved()/1e6)

        lossy = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.heads.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Only optimize the classifier layer
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Training loop
        for epoch in range(10):
            model.train()
            train_loss = 0.0
            torch.cuda.reset_peak_memory_stats()
            gc.collect(); torch.cuda.empty_cache()

            for images, labels in trainloader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(images)
                loss = lossy(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            print(f"Epoch {epoch+1} peak memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

            # Validation
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = lossy(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = 100 * correct / total
            val_loss /= len(valloader)
            train_loss /= len(trainloader)

            print(f"Epoch {epoch+1}/10 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

            scheduler.step()
            gc.collect(); torch.cuda.empty_cache()

        # test set testing
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = lossy(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_acc = 100 * correct / total
        test_loss /= len(testloader)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

        gc.collect(); torch.cuda.empty_cache()


# all the sweeps
if __name__ == "__main__":
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-5,
                'max': 1e-3
            },
            'weight_decay': {
                'min': 1e-6,
                'max': 1e-3
            },
            'batch_size': {
                'values': [128, 256]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="DL_assignment_02_part2", entity="ee22s084-indian-institute-of-technology-madras")
    wandb.agent(sweep_id, function=train_and_eval)
