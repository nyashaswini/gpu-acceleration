import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model import create_model
from optimizer import convert_to_onnx, create_onnx_session
import numpy as np

def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

def main():
    # Set up device and OpenMP threads
    device = torch.device('cpu')
    torch.set_num_threads(os.cpu_count())
    print(f"Using {torch.get_num_threads()} CPU threads")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True,
                               download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=32,
                            shuffle=True, num_workers=2)

    # Create and train model
    model = create_model(num_classes=10, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print("Starting training...")
    train(model, train_loader, criterion, optimizer, device)

    # Convert to ONNX
    print("Converting to ONNX format...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    convert_to_onnx(model, dummy_input, 'vit_model.onnx')
    
    print("Training and optimization complete!")

if __name__ == '__main__':
    main()
