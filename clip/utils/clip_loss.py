import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()

        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        assert image_embeddings.size(1) == text_embeddings.size(1), "d_model of both imgs and text need to be the same"
        
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # cosine similarity
        logits_per_image = image_embeddings @ text_embeddings.T  # (batch_size, batch_size)
        logits_per_text = logits_per_image.T  # (batch_size, batch_size)

        # scale logits by temperature
        logits_per_image /= self.temperature
        logits_per_text /= self.temperature

        # compute loss
        labels = torch.arange(logits_per_image.size(0))

        loss_image_to_text = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_text_to_image = nn.CrossEntropyLoss()(logits_per_text, labels)

        return (loss_image_to_text + loss_text_to_image) / 2

if __name__ == "__main__":
    image_embeddings = torch.randn(4, 512)  # (batch_size, embed_dim)
    text_embeddings = torch.randn(4, 512)  # (batch_size, embed_dim)

    clip_loss = CLIPLoss(temperature=0.1)
    loss = clip_loss(image_embeddings, text_embeddings)
    print(f"Loss: {loss.item():.4f}")