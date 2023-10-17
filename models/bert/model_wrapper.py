import torch
from torch import nn
from transformers import AutoModel

class ModelforSC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.model = AutoModel.from_pretrained(config.pretrained)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, config.num_classes)
        
    def forward(self, input_ids, mask):
        outputs = self.model(input_ids, attention_mask=mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if self.num_classes == 1:
            logits = torch.sigmoid(logits).squeeze(1)
        outputs = {'logits': logits}
        return outputs