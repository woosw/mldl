"""선형회귀 연습용 피처 1개 데이터 100개 생성 및 시각화."""

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt


def make_data(n: int = 100, seed: int = 42) -> tuple[list[float], list[float]]:
    random.seed(seed)
    x = [i / 10 for i in range(n)]
    y = [2.5 * value + 3 + random.gauss(0, 1.5) for value in x]
    return x, y


def save_csv(x: list[float], y: list[float], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature", "target"])
        writer.writerows(zip(x, y))


def main() -> None:
    x, y = make_data()
    csv_path = Path("linear_regression_data.csv")
    save_csv(x, y, csv_path)

    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("Feature 1")
    plt.ylabel("Target")
    plt.title("Linear Regression Training Data")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
