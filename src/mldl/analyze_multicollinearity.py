"""다중공선성 분석: 상관행렬, VIF, 조건수, 특이값."""

from pathlib import Path

import numpy as np
import pandas as pd


def calculate_vif(X: np.ndarray, names: list[str]) -> pd.DataFrame:
    values = []
    for i, name in enumerate(names):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        others = np.column_stack([np.ones(len(others)), others])
        coefficients, *_ = np.linalg.lstsq(others, y, rcond=None)
        predicted = others @ coefficients
        ss_res = np.sum((y - predicted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot
        values.append((name, 1 / (1 - r_squared)))
    return pd.DataFrame(values, columns=["feature", "VIF"])


def main() -> None:
    path = Path("multicollinearity_data.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path}가 없습니다. 먼저 multicollinearity_data.py를 실행하세요."
        )

    df = pd.read_csv(path)
    features = [column for column in df.columns if column != "target"]
    X = df[features].to_numpy(dtype=float)
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

    print(f"데이터 크기: {df.shape[0]}개 행, {len(features)}개 피처")
    print("\n[피처 간 상관행렬]")
    print(df[features].corr().round(3))

    print("\n[VIF]")
    vif = calculate_vif(X_standardized, features)
    print(vif.round(3).to_string(index=False))

    condition_number = np.linalg.cond(X_standardized)
    singular_values = np.linalg.svd(X_standardized, compute_uv=False)
    print(f"\n[조건수] {condition_number:.3f}")
    print(f"[특이값] {np.round(singular_values, 3)}")

    high_vif = vif[vif["VIF"] >= 5]["feature"].tolist()
    if high_vif:
        print(f"\n판정: 다중공선성 의심 피처 → {', '.join(high_vif)}")
        print("대응: 중복 피처 제거·통합, Ridge 회귀 또는 PCA를 검토하세요.")
    else:
        print("\n판정: VIF 기준 다중공선성이 크게 의심되지 않습니다.")


if __name__ == "__main__":
    main()
