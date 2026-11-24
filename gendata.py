import numpy as np
import pandas as pd


def generate_data(n=20000, k=3, rho=0.5, seed=42):
    assert k >= 2 and -1 <= rho <= 1

    rng = np.random.default_rng(seed)

    # 前 k-1 列独立均匀采样
    X = rng.choice([-1, 1], size=(n, k - 1))

    # 控制 k 阶依赖强度
    s = rng.choice([-1, 1], size=n, p=[(1-rho)/2, (1+rho)/2])

    # 生成最后一列
    xk = s * np.prod(X, axis=1)

    X = np.column_stack([X, xk])

    return pd.DataFrame(X, columns=[f"X{i+1}" for i in range(k)])


if __name__ == "__main__":
    # 生成三阶依赖数据
    df = generate_data(n=20000, k=3, rho=0.9)

    # 验证
    print(df.head())
    print("一阶均值:\n", df.mean())
    print("二阶相关性:\n", df.corr())
    print("高阶矩:", np.prod(df.values, axis=1).mean())

    # 保存
    df.to_csv("synthetic_data.csv", index=False)
