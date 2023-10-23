import torch
import torch.nn as nn
import math

emb = torch.load("sentence_embeddings.torch")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def mask_batch(x):
    # x shape - [batch, 9, 512]
    out = torch.clone(x)
    b = x.shape[0]
    s = x.shape[1]
    idxs = torch.randint(low=0, high=s, size=(1,b)).squeeze()
    for i, idx in enumerate(idxs):
        out[i, idx] = torch.zeros(512)
    return idxs[0], out

t = nn.Sequential(
    nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6),
    nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6),
    nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6),
)

emb = emb[:140]

def train(t, emb, epochs=100, batch_size=20, lr=1e-4):
    t.train()
    
    ds = torch.utils.data.TensorDataset(emb)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    p = PositionalEncoding(512, max_len=9)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(t.parameters(), lr=lr)

    losses = []
    accs = []
    
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        local_loss = []
        local_acc = []
        for i, batch in enumerate(dl):

            batch = batch[0]
            
            idx, masked_batch = mask_batch(batch)
            batch = p(batch)
            masked_batch = p(masked_batch)

            optimizer.zero_grad()
            
            pred = t(masked_batch)
            loss = loss_fn(pred, batch)
            
            loss.backward()
            optimizer.step()

            #print(i, loss.item())
            test_dist = nn.functional.l1_loss(pred[idx], batch[idx])

            local_loss.append(loss.item())
            local_acc.append(test_dist.item())
        losses.append(sum(local_loss) / len(local_loss))
        accs.append(sum(local_acc) / len(local_acc))

    return losses, accs