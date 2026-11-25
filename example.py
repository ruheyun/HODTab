import json
import pandas as pd
from pathlib import Path

root = Path("exp/xor")
metrics = ["rho_hat", "E1", "E2", "Ek", "TV", "E_parity", "Coverage"]

rows = []

for i in range(11):
    rho = round(i / 10, 1)

    path = root / f"{rho:.1f}" / "results2026.json"
    if not path.exists():
        path = root / f"{rho:g}" / "results2026.json"

    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    rows.append({
        "rho": rho,
        **{key: result[key] for key in metrics}
    })

df = pd.DataFrame(rows)
df.to_csv(root / "rho_metrics.csv", index=False)

print(df)
