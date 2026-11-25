from configs.base_config import get_base_config
from tabcodec import ColumnSpec


def get_config():
    config = get_base_config()
    config.seed = 2026

    cat_coder = config.training.cat_coder
    config.training.epoch = 100

    config.data.dataset = "xor"
    config.data.rho = 1
    config.data.spec = [
        ColumnSpec("X1", kind="categorical", coder=cat_coder),
        ColumnSpec("X2", kind="categorical", coder=cat_coder),
        ColumnSpec("X3", kind="categorical", coder=cat_coder),
    ]

    return config
