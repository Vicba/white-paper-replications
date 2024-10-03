import torch
import torch.nn as nn
from encoders.img_encoder import VisionEncoder
from encoders.text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, d_model=512):
        super(CLIPModel, self).__init__()
        self.vision_encoder = VisionEncoder(d_model)
        self.text_encoder = TextEncoder(d_model)

    def forward(self, images, text_tokens):
        image_embeddings = self.vision_encoder(images)
        text_embeddings = self.text_encoder(text_tokens)
        return image_embeddings, text_embeddings


if __name__ == "__main__":
    image = torch.randn(2, 3, 224, 224)
    text_tokens = torch.randint(0, 30522, (2, 77))

    model = CLIPModel()
    image_embeddings, text_embeddings = model(image, text_tokens)

    print(f"Image embeddings shape: {image_embeddings.shape}")  # should be (2, d_model)
    print(f"Text embeddings shape: {text_embeddings.shape}")  # should be (2, d_model)
