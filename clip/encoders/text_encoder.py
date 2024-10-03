import torch
import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self, d_model=512):
        super(TextEncoder, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, d_model)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        proj_embeddings = self.fc(cls_embeddings)  # (batch_size, d_model)
        return proj_embeddings


if __name__ == "__main__":
    input_ids = torch.randint(0, 30522, (2, 77))  # (batch_size, seq_len)
    
    text_encoder = TextEncoder()
    embeddings = text_encoder(input_ids)
    print(embeddings.shape)  # should be (2, d_model)
