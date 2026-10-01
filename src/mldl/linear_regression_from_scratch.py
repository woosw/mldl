"""scikit-learn 없이 선형회귀를 학습하는 예제."""

import matplotlib.pyplot as plt

from linear_regression_data import make_data


def split_data(x: list[float], y: list[float], ratio: float = 0.8):
    # 앞쪽 데이터는 학습용, 나머지는 테스트용으로 나눈다.
    cut = int(len(x) * ratio)
    return x[:cut], y[:cut], x[cut:], y[cut:]


def predict(x: list[float], weight: float, bias: float) -> list[float]:
    # 선형 모델 y = weight * x + bias로 예측한다.
    return [weight * value + bias for value in x]


def mse(actual: list[float], predicted: list[float]) -> float:
    # 예측값과 실제값의 평균 제곱 오차를 계산한다.
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)


def fit(x: list[float], y: list[float], learning_rate: float = 0.001,
        epochs: int = 3000) -> tuple[float, float, list[float]]:
    # 경사하강법으로 weight와 bias를 반복해서 갱신한다.
    weight = bias = 0.0
    losses = []

    for _ in range(epochs):
        predicted = predict(x, weight, bias)
        error = [p - actual for p, actual in zip(predicted, y)]
        # MSE의 기울기를 이용해 두 파라미터를 업데이트한다.
        weight -= learning_rate * 2 * sum(e * value for e, value in zip(error, x)) / len(x)
        bias -= learning_rate * 2 * sum(error) / len(x)
        losses.append(mse(y, predicted))

    return weight, bias, losses


def main() -> None:
    # 데이터를 만들고 학습/테스트 결과와 학습 곡선을 확인한다.
    x, y = make_data()
    train_x, train_y, test_x, test_y = split_data(x, y)
    weight, bias, losses = fit(train_x, train_y)

    print(f"모델: y = {weight:.3f}x + {bias:.3f}")
    print(f"훈련 MSE: {mse(train_y, predict(train_x, weight, bias)):.3f}")
    print(f"테스트 MSE: {mse(test_y, predict(test_x, weight, bias)):.3f}")

    plt.scatter(train_x, train_y, label="train")
    plt.scatter(test_x, test_y, label="test")
    plt.plot(x, predict(x, weight, bias), color="red", label="regression line")
    plt.xlabel("Feature 1")
    plt.ylabel("Target")
    plt.title("Linear Regression from Scratch")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
