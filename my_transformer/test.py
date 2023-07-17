import torch
import torch.utils.data as Data
from datasets import TestData, MyTranslationDataSet
from transformer import Transformer
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model:Transformer, test_data:TestData):
    # Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    
    dataset = MyTranslationDataSet(raw_data=test_data)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=1)
    for data in dataset:
        enc_input = data[0].unsqueeze(0).to(device)
        dec_input = data[1].unsqueeze(0).to(device)
        output_length = 1
        dec_next = torch.tensor([1])

        while output_length < test_data.max_length and torch.ne(dec_next[-1], 2):
            
            dec_output, enc_output = model(enc_input, dec_input)

            dec_next = dec_output.argmax().reshape(1,-1)
            print(dec_next)
            dec_input = torch.cat(dec_input[0], dec_next[-1], dim=1).unsqueeze(0)
            output_length += 1

    return dec_output


if __name__ == "__main__":
    model_path = os.getenv("USERPROFILE") + '\.cache\my_transformer\model.pth'
    test_data = TestData()

    model = Transformer(
        num_embeddings=len(test_data.tgt_dict),
        max_length=100
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device=device)
    model.eval()

    output = test(model=model, test_data=test_data)