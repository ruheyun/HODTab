import ml_collections
import torch


def get_base_config():
    config = ml_collections.ConfigDict()
    config.seed = 2026
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 256
    training.epoch = 300
    training.num_coder = 'standard'
    training.cat_coder = 'catconverter'
    training.label_coder = 'ordinal'
    
    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1000

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()

    # data
    config.data = data = ml_collections.ConfigDict()

    # model
    config.model = model = ml_collections.ConfigDict()
    
    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 1e-4

    # test
    config.test = test = ml_collections.ConfigDict()

    return config
