"""다중공선성 실습용 피처 5개, 데이터 1000개 생성."""

import csv
import math
import random
from pathlib import Path


def make_data(n: int = 1000, seed: int = 42):
    rng = random.Random(seed)
    rows = []

    for _ in range(n):
        latent = rng.gauss(0, 1)
        features = [
            latent + rng.gauss(0, 0.08),
            2 * latent + rng.gauss(0, 0.08),
            -latent + rng.gauss(0, 0.08),
            0.5 * latent + rng.gauss(0, 0.08),
            3 * latent + rng.gauss(0, 0.08),
        ]
        target = 4 * features[0] + 2 * features[1] + rng.gauss(0, 1)
        rows.append((*features, target))

    return rows


def correlation(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) *
        sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator


def save_csv(rows: list[tuple[float, ...]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "target"])
        writer.writerows(rows)


def main() -> None:
    rows = make_data()
    save_csv(rows, Path("multicollinearity_data.csv"))

    features = list(zip(*(row[:5] for row in rows)))
    print("저장 완료: multicollinearity_data.csv")
    print("피처 간 상관계수")
    for i in range(5):
        values = [correlation(features[i], features[j]) for j in range(5)]
        print(f"feature_{i + 1}: " + ", ".join(f"{value:.3f}" for value in values))


if __name__ == "__main__":
    main()
