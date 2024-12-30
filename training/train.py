import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch + 1)

    return epoch_loss, epoch_acc

def validate_one_epoch(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    writer.add_scalar('Loss/val', val_loss, epoch + 1)
    writer.add_scalar('Accuracy/val', val_acc, epoch + 1)

    return val_loss, val_acc

def train(model, train_loader, val_loader, optimizer, criterion, model_name, device, epochs=10, scheduler=None):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    if not os.path.exists(f"ckpts/{model_name}"):
        os.makedirs(f"ckpts/{model_name}", exist_ok=True)

    writer = SummaryWriter(f"runs/{model_name}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        train_epoch_loss, train_epoch_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}")

        val_epoch_loss, val_epoch_acc = validate_one_epoch(model, val_loader, criterion, device, writer, epoch)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        if scheduler is not None:
            scheduler.step(val_epoch_loss)
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.16f}")
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f"ckpts/{model_name}/best_val_loss.pth")
            print(f"Saved Best Model with Val Loss: {best_val_loss:.4f}")
        
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), f"ckpts/{model_name}/best_val_acc.pth")
            print(f"Saved Best Model with Val Acc: {best_val_acc:.4f}")

        writer.flush()

    writer.close()
    return train_loss, train_acc, val_loss, val_acc