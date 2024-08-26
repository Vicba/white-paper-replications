import torch
from model.vit import ViT

if __name__ == "__main__":
    model = ViT(image_size=224, patch_size=16, num_classes=1000, dim=1024, n_layers=6, heads=16, mlp_dim=2048)
    img = torch.randn(1, 3, 224, 224)
    preds = model(img)
    print(preds.shape)
    assert preds.shape == (1, 1000)