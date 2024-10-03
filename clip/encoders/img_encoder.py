import torch
import torch.nn as nn
from transformers import ViTModel


class VisionEncoder(nn.Module):
    def __init__(self, d_model):
        super(VisionEncoder, self).__init__()

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = nn.Linear(self.vit.config.hidden_size, d_model)

    def forward(self, images):
        # shape images: (batch_size, 3, 224, 224)
        outputs = self.vit(images) # (batch_size, hidden_size)
        image_embeddings = self.fc(outputs.last_hidden_state[:, 0, :])
        return image_embeddings


if __name__ == "__main__":
    images = torch.randn(2, 3, 224, 224)

    vision_encoder = VisionEncoder(d_model=512)
    embeddings = vision_encoder(images)
    print(embeddings.shape)  # should be (2, d_model)