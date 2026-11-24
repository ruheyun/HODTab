import pandas as pd
import torch
from tabcodec import  ColumnSpec, TabCodec
from model import TabDiff
from torch.utils.data import Dataset, DataLoader, TensorDataset


class HDataset(Dataset):
    def __init__(self, df):
        self.data = torch.tensor(df.values, dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    

def xordata():
    origin_data = pd.read_csv('synthetic_data.csv')

    spec = [
        ColumnSpec("X1",   kind="categorical", coder="catconverter"),
        ColumnSpec("X2",  kind="categorical", coder="catconverter"),
        ColumnSpec("X3", kind="categorical", coder="catconverter"),
    ]

    codec = TabCodec(spec, output='frame').fit(origin_data)

    Z = codec.transform(origin_data)

    Z_tensor = torch.tensor(Z.values, dtype=torch.float32)

    dataset = TensorDataset(Z_tensor)

    train_dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)

    diff_model = TabDiff(
        d_in=codec.output_dim,
        classes=None,
        batch_size=256,
        batch_size_sample=1000,
        device= 'cuda',
    )

    diff_model.fit(train_dataloader)

    X, y = diff_model.sample(n_samples=len(origin_data))

    reverse = codec.inverse_transform(X)   

    reverse.to_csv('reverse.csv', index=False, header=True) 

    return reverse


if __name__ == '__main__':
    xordata()