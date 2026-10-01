# 다중공선성 판별 방법

다중공선성은 여러 피처가 서로 강하게 관련되어 회귀계수 추정이 불안정해지는 문제다. 하나의 지표만 사용하기보다 상관계수, VIF, 회귀계수 안정성, 교차검증 성능을 함께 확인하는 것이 좋다.

## 1. 피처 간 상관계수 확인

두 피처 간 선형 관계를 확인한다.

| 상관계수 절댓값 | 해석 |
|---:|---|
| `0.5 미만` | 낮은 상관 |
| `0.5~0.8` | 주의 |
| `0.8 이상` | 다중공선성 의심 |
| `0.9 이상` | 매우 강한 상관 |

```python
import pandas as pd

df = pd.read_csv("multicollinearity_data.csv")
print(df.drop(columns="target").corr())
```

### 한계

상관계수는 피처 두 개씩만 비교한다. 다음과 같은 관계는 놓칠 수 있다.

$$
X_3 \approx X_1 + X_2
$$

따라서 상관계수만으로 다중공선성이 없다고 판단하면 안 된다.

## 2. VIF 확인

VIF(Variance Inflation Factor)는 실무에서 가장 대표적인 방법이다.

각 피처를 나머지 피처로 예측하는 회귀모델을 만들고, 그 설명력을 이용해 계산한다.

$$
VIF_j = \frac{1}{1-R_j^2}
$$

$R_j^2$는 피처 $X_j$를 나머지 피처로 예측한 회귀모델의 결정계수다.

| VIF | 해석 |
|---:|---|
| `1` | 다른 피처와 거의 관계 없음 |
| `1~5` | 일반적으로 허용 가능 |
| `5 이상` | 다중공선성 주의 |
| `10 이상` | 심각한 다중공선성 의심 |

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("multicollinearity_data.csv")
X = df.drop(columns="target")

vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ],
})

print(vif)
```

VIF는 여러 피처가 결합되어 만드는 다중공선성도 확인할 수 있기 때문에 단순 상관계수보다 유용하다.

## 3. 회귀계수의 불안정성 확인

다중공선성이 있으면 데이터를 조금만 변경해도 회귀계수가 크게 변할 수 있다.

예를 들어 데이터 분할에 따라 다음처럼 계수의 크기와 부호가 바뀔 수 있다.

```text
feature_1 계수:  4.2  → -12.4
feature_2 계수:  2.1  →  18.7
feature_3 계수: -0.8  →   5.3
```

예측 성능이 반드시 크게 나빠지는 것은 아니지만, 다음 문제가 발생할 수 있다.

- 개별 피처의 영향 해석이 어려워짐
- 회귀계수가 불안정해짐
- 계수의 부호가 직관과 다르게 나타남
- 표준오차가 커짐
- 통계적 유의성이 낮아짐

## 4. 표준오차와 p-value 확인

통계적 선형회귀에서는 회귀계수의 표준오차가 커지는지 확인할 수 있다.

모델 전체의 설명력은 높은데 개별 피처의 p-value가 높다면 다중공선성을 의심할 수 있다.

```text
R² = 0.95
feature_1 p-value = 0.42
feature_2 p-value = 0.67
feature_3 p-value = 0.51
```

다만 p-value만으로 다중공선성을 판정하면 안 된다. 표본 크기와 오차 분산의 영향도 함께 받기 때문이다.

## 5. 조건수 확인

설계행렬 $X$의 조건수도 다중공선성을 확인하는 방법이다.

$$
\kappa(X) = \frac{\sigma_{max}}{\sigma_{min}}
$$

여기서 $\sigma_{max}$와 $\sigma_{min}$은 특이값의 최댓값과 최솟값이다.

| 조건수 | 해석 |
|---:|---|
| 낮음 | 문제 가능성 낮음 |
| 수백 이상 | 주의 |
| 수천 이상 | 심각한 다중공선성 가능성 |

```python
import numpy as np

X = df.drop(columns="target")
condition_number = np.linalg.cond(X)
print(condition_number)
```

피처의 단위가 서로 크게 다르면 조건수가 왜곡될 수 있으므로 표준화한 뒤 확인하는 것이 좋다.

## 6. 고유값과 특이값 확인

피처 행렬의 고유값 또는 특이값 중 0에 가까운 값이 있다면 피처 사이에 강한 선형 종속 관계가 있을 수 있다.

```python
import numpy as np

X = df.drop(columns="target")
X = (X - X.mean()) / X.std()

singular_values = np.linalg.svd(X, compute_uv=False)
print(singular_values)
```

특이값 중 일부가 매우 작다면 피처 행렬이 거의 중복된 정보를 포함한다는 의미다.

## 추천 판별 절차

### 1단계: 결측값과 데이터 타입 확인

```python
print(df.isna().sum())
print(df.dtypes)
```

### 2단계: 피처 표준화

피처마다 단위가 다르면 VIF나 조건수 해석에 영향을 줄 수 있다.

### 3단계: 상관행렬 확인

```python
print(X.corr())
```

절댓값이 `0.8` 이상인 피처 쌍을 우선 확인한다.

### 4단계: VIF 계산

VIF가 높은 피처를 확인한다.

### 5단계: 모델을 여러 번 학습

훈련 데이터 분할이나 교차검증을 바꿔가며 계수가 안정적인지 확인한다.

### 6단계: 목적에 따라 처리

- 예측이 목적이면 정규화 회귀를 고려한다.
- 변수 해석이 목적이면 중복 피처를 제거하거나 통합한다.
- 중요한 정보를 잃고 싶지 않으면 차원축소를 고려한다.

## 다중공선성 해결 방법

### 1. 중복 피처 제거

상관이 매우 높은 피처 중 하나를 제거한다. 도메인 지식, 결측률, 측정 품질 등을 고려해 선택한다.

### 2. 피처 통합

비슷한 피처를 평균이나 합계로 통합한다.

```python
df["combined_feature"] = (
    df["feature_1"] + df["feature_2"]
) / 2
```

### 3. Ridge 회귀 사용

Ridge 회귀는 계수에 패널티를 적용해 계수의 불안정성을 줄인다.

$$
\min_{\beta}
\left[
\sum (y_i-\hat{y}_i)^2
+ \lambda\sum\beta_j^2
\right]
$$

### 4. Lasso 회귀 사용

Lasso는 일부 계수를 0으로 만들어 피처 선택 효과를 낸다.

$$
\min_{\beta}
\left[
\sum (y_i-\hat{y}_i)^2
+ \lambda\sum|\beta_j|
\right]
$$

서로 강하게 상관된 피처 중 어떤 피처를 선택할지는 데이터에 따라 불안정할 수 있다.

### 5. PCA 사용

상관된 피처들을 서로 독립적인 주성분으로 변환한다. 단, 주성분은 원래 피처와 직접적인 의미가 약해질 수 있다.

## 중요한 주의점

다중공선성이 항상 제거해야 할 문제는 아니다.

예측이 목적이라면 상관된 피처들이 예측에 도움이 될 수 있다. 다중공선성의 가장 큰 문제는 모델의 개별 계수를 해석하거나 원인과 영향을 설명하려 할 때 발생하는 불안정성이다.

목적별로 다음 지표를 우선한다.

| 목적 | 우선 확인할 항목 |
|---|---|
| 예측 중심 | 교차검증 성능, RMSE, MAE |
| 해석 중심 | VIF, 계수 안정성, 표준오차 |
| 변수 선택 중심 | 상관행렬, VIF, Lasso |

## 핵심 정리

가장 추천하는 조합은 다음과 같다.

```text
상관행렬
+ VIF
+ 회귀계수 안정성
+ 교차검증 성능
```

상관계수만으로는 부족하다. 특히 다중공선성 판별에는 VIF가 가장 실용적인 출발점이다.
