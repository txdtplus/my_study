# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import torch.nn as nn
import torch.optim as optim
from beifen.datasets import *
from transformer_old import Transformer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    enc_inputs, dec_inputs, dec_outputs = make_data()
    dataset = MyDataSet(enc_inputs=enc_inputs, dec_inputs=dec_inputs, dec_outputs=dec_outputs)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)

    model = Transformer()
    model.to(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in dataloader:  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)
            
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    write_folder = os.getenv("USERPROFILE") + '\.cache\my_transformer'
    if not os.path.exists(write_folder):
        os.mkdir(write_folder)

    
    torch.save(model.state_dict(), os.path.join(write_folder, 'model.pth'))
    print("保存模型")
