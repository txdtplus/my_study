import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=5, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()        # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                  # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())


if __name__ == "__main__":
    d_model = 5
    max_len = 10
    model = PositionalEncoding(d_model=d_model, max_len=max_len)
    pos_table = model.pos_table.cpu().numpy()
    tmp = np.array(
        [
            [pos / np.power(10000, 2*i / d_model) for i in range(d_model)] 
            for pos in range(max_len)
        ]
    )

    a = tmp.copy()

    a[1:, 0::2] = np.sin(tmp[1:, 0::2])
    a[1:, 1::2] = np.cos(tmp[1:, 1::2])

    print(pos_table[1] - a[1])