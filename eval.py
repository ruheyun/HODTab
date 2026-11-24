import numpy as np
from itertools import product, combinations


def evaluate(df_syn, rho):
    X = df_syn.to_numpy()
    n, k = X.shape

    assert -1 <= rho <= 1
    assert np.isin(X, [-1, 1]).all()

    # 1. 一阶边缘分布误差
    E1 = np.abs((X == 1).mean(axis=0) - 0.5).mean()

    # 2. 二阶联合分布误差（平均 TV）
    E2 = np.mean([
        0.5 * sum(
            abs(np.mean((X[:, i] == a) & (X[:, j] == b)) - 0.25)
            for a, b in product([-1, 1], repeat=2)
        )
        for i, j in combinations(range(k), 2)
    ])

    # 3. 高阶矩及其误差
    prod = np.prod(X, axis=1)
    rho_hat = prod.mean()
    Ek = abs(rho_hat - rho)

    # 4. 整体联合分布 TV 距离
    states = list(product([-1, 1], repeat=k))
    p_real = np.array([(1 + rho * np.prod(s)) / 2**k for s in states])
    p_syn = np.array([np.mean(np.all(X == s, axis=1)) for s in states])
    TV = 0.5 * np.abs(p_real - p_syn).sum()

    # 5. 奇偶约束概率误差
    E_parity = abs(np.mean(prod == 1) - (1 + rho) / 2)

    # 6. 模式覆盖率（真实分布支持集）
    coverage = np.mean(p_syn[p_real > 1e-12] > 0)

    return {
        "E1": E1,
        "E2": E2,
        "rho_hat": rho_hat,
        "Ek": Ek,
        "TV": TV,
        "E_parity": E_parity,
        "Coverage": coverage,
    }
