from configs.base_config import get_base_config
from tabcodec import ColumnSpec


def get_config():
    config = get_base_config()

    num_coder = config.training.num_coder
    cat_coder = config.training.cat_coder
    label_coder = config.training.label_coder

    config.data.dataset = "default"
    config.data.spec = [
        ColumnSpec("LIMIT_BAL", kind="numeric", coder=num_coder),
        ColumnSpec("BILL_AMT1", kind="numeric", coder=num_coder),
        ColumnSpec("BILL_AMT2", kind="numeric", coder=num_coder),
        ColumnSpec("BILL_AMT3", kind="numeric", coder=num_coder),
        ColumnSpec("BILL_AMT4", kind="numeric", coder=num_coder),
        ColumnSpec("BILL_AMT5", kind="numeric", coder=num_coder),
        ColumnSpec("BILL_AMT6", kind="numeric", coder=num_coder),
        ColumnSpec("PAY_AMT1", kind="numeric", coder=num_coder),
        ColumnSpec("PAY_AMT2", kind="numeric", coder=num_coder),
        ColumnSpec("PAY_AMT3", kind="numeric", coder=num_coder),
        ColumnSpec("PAY_AMT4", kind="numeric", coder=num_coder),
        ColumnSpec("PAY_AMT5", kind="numeric", coder=num_coder),
        ColumnSpec("PAY_AMT6", kind="numeric", coder=num_coder),
        ColumnSpec("SEX", kind="categorical", coder=cat_coder),
        ColumnSpec("EDUCATION", kind="categorical", coder=cat_coder),
        ColumnSpec("MARRIAGE", kind="categorical", coder=cat_coder),
        ColumnSpec("AGE", kind="categorical", coder=cat_coder),
        ColumnSpec("PAY_0", kind="categorical", coder=cat_coder),
        ColumnSpec("PAY_1", kind="categorical", coder=cat_coder),
        ColumnSpec("PAY_2", kind="categorical", coder=cat_coder),
        ColumnSpec("PAY_3", kind="categorical", coder=cat_coder),
        ColumnSpec("PAY_4", kind="categorical", coder=cat_coder),
        ColumnSpec("PAY_5", kind="categorical", coder=cat_coder),
        ColumnSpec("Label", kind="categorical", coder=label_coder),
    ]

    return config
