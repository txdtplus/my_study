import torch
import torch.nn as nn
import torch.nn.functional as F
 
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):
    def __init__(self, ) -> None:
        super(Transformer, self).__init__()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_length=6):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.position_encoding = self.create_position_encoding(d_model, max_length)
 
    def create_position_encoding(self, d_model, max_length):
        position_encoding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2, dtype=float) * math.log(10000) / d_model)
        tmp = position * div_term
        position_encoding[:, 0::2] = torch.sin(tmp)
        position_encoding[:, 1::2] = torch.cos(tmp)
        return position_encoding.unsqueeze(0)
    
    def forward(self, x:torch.Tensor):
        return x + self.position_encoding[:, :x.size(1), :].to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=6):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dk = d_model // num_heads
        self.dv = d_model // num_heads

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):
        # input: Q: [batch_size, sentence_length, d_model]

        batch_size = Q.size(0)

        query = self.query_linear(Q).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)      # [batch_size, num_heads, sentence_length, dk]
        key = self.key_linear(K).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)          # [batch_size, num_heads, sentence_length, dk]
        value = self.value_linear(V).view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)

        # value2 = self.value_linear(V).view(batch_size, self.num_heads, -1, self.dk)
        # 由于num_heads和dk是将d_model扩展成两个维度，所以这两个放在后面，然后转置
        # 因此value2这种写法不对

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dk)
        attention = torch.matmul(F.softmax(scores, dim=-1), value)

        return value, value2

        


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

if __name__ == "__main__":
    x = torch.tensor([[1,2,3], [4,5,6]]).to(device)
    d_model = 12

    layer = nn.Embedding(num_embeddings=10, embedding_dim=d_model)
    position_encoding = PositionalEncoding(d_model=d_model)

    layer.to(device)
    position_encoding.to(device)

    x = layer(x)
    print(x.shape)
    y = position_encoding(x)
    print(y.shape)

    mha = MultiHeadAttention(d_model=d_model)
    mha.to(device=device)
    value, value2 = mha(y,y,y)

    print(value - value2)
    