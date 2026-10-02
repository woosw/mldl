import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# 1. 한글 폰트 설정 (Windows 환경)
# -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------------
# 2. 샘플 데이터 생성 (1,000개)
# -------------------------------------------------------------------
# 재현성을 위한 시드 설정
np.random.seed(42)

n_samples = 1000
# -3에서 3 사이의 X값 균등 분포 생성
X = np.sort(np.random.uniform(-3, 3, size=(n_samples, 1)), axis=0)

# 실제 생성 함수 (3차 다항식): y = 0.5 * x^3 - 1.2 * x^2 + 0.8 * x + 3
true_y = 0.5 * (X ** 3) - 1.2 * (X ** 2) + 0.8 * X + 3

# 노이즈(오차항) 추가 (정규분포 평균 0, 표준편차 2.0)
noise = np.random.normal(0, 2.0, size=(n_samples, 1))
y = true_y + noise

# 데이터프레임으로 변환 및 CSV 저장 (필요 시 활용)
df = pd.DataFrame({'x': X.flatten(), 'y': y.flatten()})
df.to_csv("polynomial_sample_data.csv", index=False)
print("1,000개의 샘플 데이터가 'polynomial_sample_data.csv'로 저장되었습니다.")
print(df.head())

# -------------------------------------------------------------------
# 3. 데이터 분할 (학습용 80%, 테스트용 20%)
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------
# 4. 다항 선형회귀 모델 학습 (차수별 비교: 1차, 2차, 3차)
# -------------------------------------------------------------------
# 시각화를 위한 촘촘한 X 축 데이터
X_plot = np.linspace(-3, 3, 500).reshape(-1, 1)
y_plot_true = 0.5 * (X_plot ** 3) - 1.2 * (X_plot ** 2) + 0.8 * X_plot + 3

degrees = [1, 2, 3]
models = {}
predictions = {}
metrics = {}

for deg in degrees:
    # 다항 특성 변환 (X -> [1, X, X^2, ..., X^deg])
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    X_plot_poly = poly.transform(X_plot)

    # 선형 회귀 모델 적합
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # 평가 및 예측
    y_pred_test = model.predict(X_test_poly)
    y_pred_plot = model.predict(X_plot_poly)

    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    models[deg] = model
    predictions[deg] = y_pred_plot
    metrics[deg] = (mse, r2)
    print(f"Degree {deg} -> MSE: {mse:.4f}, R² Score: {r2:.4f}")

# -------------------------------------------------------------------
# 5. 그래프 시각화
# -------------------------------------------------------------------
plt.figure(figsize=(12, 7))

# (1) 샘플 데이터 산점도 (1,000개 중 가독성을 위해 train/test를 구분하거나 전체 표시)
plt.scatter(X, y, color='lightgray', edgecolors='gray', alpha=0.5, s=25, label='샘플 데이터 (1,000개)')

# (2) 실제 참(True) 함수 곡선
plt.plot(X_plot, y_plot_true, color='black', linestyle='--', linewidth=2.5, label='실제 함수 곡선 (True Function)')

# (3) 다항 회귀 예측선 (1차, 2차, 3차)
colors = ['tab:red', 'tab:green', 'tab:blue']
styles = ['-.', ':', '-']
for deg, color, style in zip(degrees, colors, styles):
    mse, r2 = metrics[deg]
    plt.plot(
        X_plot,
        predictions[deg],
        color=color,
        linestyle=style,
        linewidth=2,
        label=f'다항회귀 Degree {deg} (R²={r2:.3f}, MSE={mse:.2f})'
    )

plt.title('다항 선형 회귀 (Polynomial Linear Regression) 샘플 데이터 및 차수별 적합선', fontsize=15, pad=12)
plt.xlabel('X (특성값)', fontsize=12)
plt.ylabel('y (타겟값)', fontsize=12)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 그래프 저장 및 출력
output_image_path = "polynomial_regression_result.png"
plt.savefig(output_image_path, dpi=300)
print(f"결과 그래프가 '{output_image_path}' 파일로 저장되었습니다.", flush=True)

# GUI 창 띄우기 (스크립트를 직접 실행할 때 창 표시)
# 터미널 환경 등에서 창을 즉시 닫고 싶다면 plt.close()를 사용할 수 있습니다.
plt.show()

