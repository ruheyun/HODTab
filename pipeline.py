import pandas as pd
import numpy as np
import torch
import json
import os
from pathlib import Path
from tabcodec import TabCodec
from model import TabDiff
from torch.utils.data import DataLoader, TensorDataset
from data import get_data
from absl import flags, app
from ml_collections.config_flags import config_flags
from eval import evaluate

    
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", 'configs/xor.py', "Training configuration.", lock_config=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(argv):

    randomSeed = FLAGS.config.seed
    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(randomSeed)

    data_name = FLAGS.config.data.dataset
    rho = FLAGS.config.data.rho
    spec = FLAGS.config.data.spec

    origin_X, origin_y = get_data(data_name=data_name, rho=rho)

    codec = TabCodec(spec, output='frame').fit(origin_X)

    Z = codec.transform(origin_X)

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

    X, y = diff_model.sample(n_samples=len(origin_X))

    reverse = codec.inverse_transform(X)   

    out_put = os.path.join('exp', data_name, str(rho), f'results{randomSeed}.json')


    result = evaluate(reverse, rho)

    write_json(out_put, {
        **result,
    })

    reverse.to_csv(f'exp/{data_name}/{str(rho)}/reverse{randomSeed}.csv', index=False, header=True) 
    print(f'Results saved to {out_put}')


if __name__ == '__main__':
    app.run(main)
