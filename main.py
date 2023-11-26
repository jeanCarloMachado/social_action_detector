
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BinaryClassification().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

# END: abpxx6d04wxr
