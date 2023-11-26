
import torch
import torch.nn as nn
from transformers import AutoModel

class BinaryClassification(nn.Module):
   def __init__(self):
       super(BinaryClassification, self).__init__()
       self.base_model = AutoModel.from_pretrained('bert-base-uncased')
       self.dropout = nn.Dropout(0.5)
       self.linear = nn.Linear(768, 2)


   def forward(self, input_ids, attn_mask):
       outputs = self.base_model(input_ids, attention_mask=attn_mask)
       outputs = self.dropout(outputs[0])
       outputs = self.linear(outputs)

       return outputs