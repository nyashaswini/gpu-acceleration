import torch
import torch.nn as nn
import timm

class SimpleViT(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        # Using a smaller ViT model for CPU training
        self.model = timm.create_model('vit_tiny_patch16_224', 
                                     pretrained=pretrained,
                                     num_classes=num_classes)
        
        # Enable OpenMP optimizations for linear layers
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                setattr(m, 'num_threads', torch.get_num_threads())

    def forward(self, x):
        return self.model(x)

def create_model(num_classes=10, pretrained=False):
    model = SimpleViT(num_classes=num_classes, pretrained=pretrained)
    return model
