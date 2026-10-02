# 와인 데이터셋을 활용한 다중 클래스 로지스틱 회귀 (Multiclass Logistic Regression) 완벽 가이드

이 문서는 [`wine_logistic_regression.py`](file:///C:/Users/sun/orca/workspaces/mldl/corbina/wine_logistic_regression.py) 코드의 머신러닝/통계학적 배경 이론, 데이터 전처리 파이프라인의 필요성, 코드 단계별 상세 설명 및 시각화 결과 해석을 체계적으로 정리한 가이드입니다.

---

## 📌 목차
1. [데이터셋 구조 및 특성 분석](#1-데이터셋-구조-및-특성-분석)
2. [핵심 이론 및 엔지니어링 배경](#2-핵심-이론-및-엔지니어링-배경)
   - 왜 분류 문제인데 '회귀(Regression)'인가?
   - 이진 시그모이드에서 다중 소프트맥스(Softmax)로의 확장
   - 교차 엔트로피 손실 함수(Cross-Entropy Loss / Log-Loss)
   - 왜 `StandardScaler` 전처리가 필수인가?
3. [전체 소스 코드](#3-전체-소스-코드)
4. [단계별 코드 상세 해설](#4-단계별-코드-상세-해설)
5. [평가 지표 및 시각화 결과 해석](#5-평가-지표-및-시각화-결과-해석)
6. [30년 차 데이터 엔지니어의 실무 요약 노트](#6-30년-차-데이터-엔지니어의-실무-요약-노트)

---

## 1. 데이터셋 구조 및 특성 분석

* **출처**: Scikit-Learn 내장 Toy Dataset (`from sklearn.datasets import load_wine`)
* **데이터 규모**: 총 178개 샘플(와인 표본), 13개 연속형 특성
* **타겟 클래스**: 3가지 재배 품종 (이탈리아 같은 지역에서 3개 다른 품종으로 만든 와인)
  * `Class_0`: 59개 (33.1%)
  * `Class_1`: 71개 (39.9%)
  * `Class_2`: 48개 (27.0%)
* **주요 화학 분석 특성**:
  * `alcohol` (알코올 도수)
  * `malic_acid` (말산 함량)
  * `total_phenols` / `flavanoids` (페놀류 및 플라보노이드 항산화 성분)
  * `color_intensity` (색상 강도)
  * `proline` (아미노산의 일종인 프롤린 함량)

---

## 2. 핵심 이론 및 엔지니어링 배경

### (1) 왜 이름이 '로지스틱 회귀(Regression)'인가?
분류(Classification) 모델임에도 '회귀'라는 단어가 붙은 이유는, **입력값의 선형 결합(선형 회귀식)**을 먼저 계산한 뒤 이를 확률로 사상(Mapping)하기 때문입니다.

1. **선형 결합 계산**: $z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p = \mathbf{w}^T \mathbf{x}$
2. **비선형 활성화 함수 변환**: 선형 회귀의 출력 $z \in (-\infty, \infty)$를 확률값 $P \in [0, 1]$로 압축합니다.

### (2) 시그모이드(Sigmoid)에서 소프트맥스(Softmax)로
* **이진 로지스틱 회귀 (2개 클래스)**:
  $$\sigma(z) = \frac{1}{1 + e^{-z}} = P(y=1|\mathbf{x})$$
* **다중 로지스틱 회귀 (Multiclass, $K$개 클래스)**:
  각 클래스 $k$에 대해 선형 결합 $z_k = \mathbf{w}_k^T \mathbf{x}$를 구한 뒤, 모든 클래스 값의 지수합으로 나누어 **전체 클래스 확률의 합이 정확히 1.0**이 되도록 정규화합니다.
  $$P(y = k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

### (3) 손실 함수: 다중 교차 엔트로피 (Multiclass Cross-Entropy / Log-Loss)
모델은 정답 클래스에 예측한 확률 $P(y_i = y_i^*)$가 $1.0$에 가까워지도록 아래 손실(Loss)을 최소화합니다:
$$\mathcal{L}_{\text{log-loss}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log\left(P(y_i = k | \mathbf{x}_i)\right)$$
* 만약 실제 정답이 Class 1인데 모델이 Class 1의 확률을 0.99로 예측하면 손실은 거의 0입니다.
* 반대로 정답이 Class 1인데 모델이 Class 1의 확률을 0.01로 예측하면 $-\log(0.01) \approx 4.6$의 막대한 페널티(벌점)를 부여받아 가중치가 크게 업데이트됩니다.

### (4) 왜 `StandardScaler`가 절대적인 필수인가?
Wine 데이터셋의 두 특성을 비교해 보면 다음과 같습니다:
* `proline`: 평균 약 **746.9**, 최대 **1680.0**
* `nonflavanoid_phenols`: 평균 약 **0.36**, 최대 **0.66**

두 변수 사이의 단위 스케일이 무려 **2,000배 이상** 차이 납니다. 스케일링을 하지 않으면:
1. **경사하강법/L-BFGS 수렴 실패**: 등고선이 길쭉한 타원형이 되어 최적해를 찾는 데 매우 오랜 시간이 걸리거나 진동(Oscillation)합니다.
2. **부당한 규제($L_2$ Penalty) 왜곡**: Scikit-Learn의 `LogisticRegression`은 기본적으로 $L_2$ 규제(`C=1.0`)가 적용되어 있습니다. 값이 큰 `proline`의 가중치만 과도하게 억압받고, 작은 값의 특성은 무시되는 심각한 왜곡이 발생합니다.
* 따라서 모든 특성을 $\mu=0, \sigma=1$로 맞추는 **표준정규화(StandardScaler)**는 필수 전처리입니다.

---

## 3. 전체 소스 코드

[wine_logistic_regression.py](file:///C:/Users/sun/orca/workspaces/mldl/corbina/wine_logistic_regression.py) 파일 내용:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# 1. 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 로드 및 탐색
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
target_names = [f"Class_{i} ({name})" for i, name in enumerate(wine.target_names)]
feature_names = wine.feature_names

# 3. 데이터 분할 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. 전처리 + 다중 클래스 로지스틱 회귀 파이프라인
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
model_pipeline.fit(X_train, y_train)

# 5. 모델 평가
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)

print(f"★ 모델 테스트 정확도(Accuracy): {acc * 100:.2f}%")
print(f"★ 다중 클래스 교차 엔트로피 손실(Log-Loss): {loss:.4f}\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. 종합 시각화 (4분면 대시보드)
fig = plt.figure(figsize=(18, 12))

# (1) 스케일링 전/후 비교
ax1 = plt.subplot(2, 2, 1)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
sample_features = ['alcohol', 'magnesium', 'proline', 'nonflavanoid_phenols']

b1 = ax1.boxplot([X[c] for c in sample_features], positions=np.arange(4)*2.0 - 0.35, widths=0.5,
                 patch_artist=True, boxprops=dict(facecolor='lightcoral'))
b2 = ax1.boxplot([X_scaled[c] for c in sample_features], positions=np.arange(4)*2.0 + 0.35, widths=0.5,
                 patch_artist=True, boxprops=dict(facecolor='cornflowerblue'))
ax1.set_xticks(np.arange(4)*2.0)
ax1.set_xticklabels(sample_features, fontsize=10, rotation=15)
ax1.set_yscale('symlog')
ax1.set_title('(1) StandardScaler 전/후 특성 분포 비교 (스케일 정규화)', fontsize=13)
ax1.legend([b1["boxes"][0], b2["boxes"][0]], ['스케일링 전 (Raw)', '표준화 후 (StandardScaled)'])
ax1.grid(True, linestyle=':', alpha=0.6)

# (2) 혼동 행렬
ax2 = plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class_0', 'Class_1', 'Class_2'])
disp.plot(cmap='Blues', ax=ax2, colorbar=False)
ax2.set_title(f'(2) 혼동 행렬 (Confusion Matrix) - 정확도: {acc*100:.1f}%', fontsize=13)

# (3) 소프트맥스 예측 확률
ax3 = plt.subplot(2, 2, 3)
sample_indices = np.arange(min(15, len(y_test)))
bottom_vals = np.zeros(len(sample_indices))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for c in range(3):
    probs = y_prob[sample_indices, c]
    ax3.bar(sample_indices, probs, bottom=bottom_vals, label=f'P({target_names[c]})', color=colors[c], width=0.65)
    bottom_vals += probs
ax3.set_xticks(sample_indices)
ax3.set_xticklabels([f"#{i}\n(정답:{y_test.iloc[i]})" for i in sample_indices], fontsize=9)
ax3.set_title('(3) 테스트 샘플별 소프트맥스(Softmax) 클래스 예측 확률 (합=1.0)', fontsize=13)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(axis='y', linestyle=':', alpha=0.6)

# (4) PCA 2D 결정 경계면
ax4 = plt.subplot(2, 2, 4)
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(StandardScaler().fit_transform(X_train))
X_test_pca = pca.transform(StandardScaler().fit_transform(X_test))

clf_2d = LogisticRegression(random_state=42).fit(X_train_pca, y_train)
xx, yy = np.meshgrid(np.linspace(X_train_pca[:,0].min()-1, X_train_pca[:,0].max()+1, 300),
                     np.linspace(X_train_pca[:,1].min()-1, X_train_pca[:,1].max()+1, 300))
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax4.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.coolwarm)

for i, c_name in enumerate(['Class_0', 'Class_1', 'Class_2']):
    ax4.scatter(X_test_pca[y_test == i, 0], X_test_pca[y_test == i, 1], edgecolors='k', s=60, label=f'실제 {c_name}')
ax4.set_title('(4) PCA 2D 투영 평면에서의 로지스틱 회귀 다중 결정 경계', fontsize=13)
ax4.legend(loc='lower left', fontsize=10)
ax4.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("wine_logistic_regression_result.png", dpi=300)
```

---

## 4. 단계별 코드 상세 해설

### Step 1: 데이터 로드 및 층화 추출 (`stratify=y`)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```
* `stratify=y`: 와인 데이터셋은 3개 품종의 비율이 59:71:48로 균등하지 않습니다. 무작위 분할 시 특정 클래스가 테스트 세트에 과소/과대 포함되는 왜곡을 방지하기 위해, 원본 데이터의 클래스 비율을 학습/테스트 세트에 동일하게 보존합니다.

### Step 2: 전처리-모델 파이프라인 구성 (`Pipeline`)
```python
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
```
* **데이터 누수(Data Leakage) 원천 차단**: 테스트 세트의 평균과 분산이 학습 과정에 새어 들어가지 않도록 `Pipeline` 객체를 사용합니다. `fit()` 시점에는 오직 학습 데이터(`X_train`)의 통계량으로만 스케일러가 기준을 잡고, 테스트 시에는 그 기준으로 변환만 수행합니다.

### Step 3: 확률 예측과 교차 엔트로피 손실 측정
```python
y_pred = model_pipeline.predict(X_test)        # 최종 라벨 예측 (0, 1, 2)
y_prob = model_pipeline.predict_proba(X_test)  # 소프트맥스 확률 벡터 ([p0, p1, p2])
loss = log_loss(y_test, y_prob)               # Cross-Entropy 손실 계산
```
* `predict_proba()`는 각 샘플마다 $[P(C_0), P(C_1), P(C_2)]$ 3개 확률값의 합이 1.0인 배열을 반환합니다.
* `log_loss`를 측정하여 모델이 정답 클래스에 얼마나 높은 신뢰도(Confidence)로 확률을 부여했는지를 연속적인 손실값으로 검증합니다.

---

## 5. 평가 지표 및 시각화 결과 해석

![시각화 결과](file:///C:/Users/sun/orca/workspaces/mldl/corbina/wine_logistic_regression_result.png)

### (1) 모델 정량 평가 수치
```text
★ 모델 테스트 정확도(Accuracy): 100.00%
★ 다중 클래스 교차 엔트로피 손실(Log-Loss): 0.0671

[상세 분류 리포트 (Classification Report)]
                   precision    recall  f1-score   support
Class_0 (class_0)       1.00      1.00      1.00        15
Class_1 (class_1)       1.00      1.00      1.00        18
Class_2 (class_2)       1.00      1.00      1.00        12
         accuracy                           1.00        45
```
* **정확도 100%**: 45개 테스트 데이터에 대해 단 1건의 오분류도 발생하지 않았습니다.
* **Log-Loss 0.0671**: 단순 정답 여부뿐 아니라, 소프트맥스 확률 예측값이 정답 클래스에 대해 95~99% 이상으로 매우 확신 있게 예측되었음을 의미합니다.

### (2) 4분면 시각화 대시보드 해설
1. **(1) 스케일링 전/후 박스플롯 (좌상단)**:
   * 붉은색(Raw) 상태에서는 `proline`이 1,000 수준인데 반해 `nonflavanoid_phenols`는 0.3 부근에 납작하게 붙어 있습니다.
   * 파란색(StandardScaled)으로 변환 후에는 모든 변수가 평균 0, 편차 1 수준으로 표준화되어 대칭적인 분포를 갖추게 됩니다.
2. **(2) 혼동 행렬 (우상단)**:
   * 주대각선(15, 18, 12)에만 샘플이 배치되어 완벽한 분류 성능을 한눈에 입증합니다.
3. **(3) 샘플별 소프트맥스 확률 누적 막대 (좌하단)**:
   * 각 테스트 와인 표본마다 파랑($C_0$), 주황($C_1$), 초록($C_2$) 영역의 높이 합이 정확히 1.0(100%)을 이룹니다.
   * 대부분의 막대가 단일 색상으로 95% 이상 꽉 차 있어, 로지스틱 회귀 모델이 매우 뚜렷한 화학적 경계선으로 품종을 구별해 냈음을 보여줍니다.
4. **(4) PCA 2D 결정 경계면 (우하단)**:
   * 13개의 다차원 특성을 2개의 주성분 평면으로 압축했을 때, 로지스틱 회귀가 공간을 **3개의 선형 분할 영역(붉은색, 푸른색, 회색 영역)**으로 구분하여 데이터를 깨끗하게 갈라놓은 모습을 확인할 수 있습니다.

---

## 6. 30년 차 데이터 엔지니어의 실무 요약 노트

1. **로지스틱 회귀는 선형 분류기다**:
   * 아무리 다중 클래스이고 소프트맥스를 쓴다 해도, 결정 경계(Decision Boundary) 자체는 항상 **선형(Linear Hyperplane)**입니다. 만약 데이터가 원형이나 비선형으로 꼬여 있다면 커널 서포트 벡터 머신(SVM)이나 신경망(MLP)으로 넘어가야 합니다.
2. **Scikit-Learn의 LogisticRegression 기본 동작을 기억하라**:
   * `LogisticRegression`은 기본적으로 $L_2$ 규제(`penalty='l2'`, `C=1.0`)가 켜져 있습니다. 따라서 **특성 스케일링(`StandardScaler`) 없이 투입하면 모델이 엉망으로 학습**됩니다.
3. **실무 파이프라인의 모범 답안은 언제나 `Pipeline`이다**:
   * 스케일러와 분류기를 따로따로 돌리지 말고 `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])` 형태로 묶는 습관이 데이터 누수(Data Leakage)와 운영 배포 시 실수를 원천 방지합니다.
