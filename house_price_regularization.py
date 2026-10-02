import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

# -------------------------------------------------------------------
# 1. 환경 설정 및 한글 폰트 지정
# -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------------
# 2. 데이터 로드 (지정된 경로)
# -------------------------------------------------------------------
train_path = "data/house_train.csv"
test_path = "data/house_test.csv"
submission_path = "data/house_sample_submission.csv"

print(">> [1/5] 데이터셋 로드 중...", flush=True)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(submission_path)

print(f" - Train Shape: {train_df.shape}")
print(f" - Test Shape:  {test_df.shape}")
print(f" - Submission Shape: {sample_sub.shape}")

# 특성과 타겟 분리 (집값 데이터는 왜도가 심하므로 log1p 변환이 표준)
X = train_df.drop(columns=['Id', 'SalePrice'])
y_log = np.log1p(train_df['SalePrice'])
X_test_raw = test_df.drop(columns=['Id'])

# -------------------------------------------------------------------
# 3. 데이터 전처리 파이프라인 (Data Pipeline)
# -------------------------------------------------------------------
print(">> [2/5] 전처리 파이프라인 구축 및 특성 인코딩...", flush=True)
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

# 수치형: 결측치 중앙값 대체 + 표준화(StandardScaler는 L1/L2 규제에 필수)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 범주형: 결측치 'Missing' 대체 + One-Hot Encoding
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
)

# 교차 검증 및 평가를 위한 Train/Validation 분할 (8:2)
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# 전처리 적합 및 변환
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test_raw)

# 전체 특성 이름 추출 (회귀 계수 분석용)
encoded_cat_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
feature_names = np.array(num_cols + list(encoded_cat_names))
print(f" - 인코딩 완료 후 총 특성 수: {len(feature_names)}개", flush=True)

# -------------------------------------------------------------------
# 4. Ridge, Lasso, ElasticNet 교차 검증 학습
# -------------------------------------------------------------------
print(">> [3/5] Ridge, Lasso, ElasticNet 교차검증(CV) 학습 시작...", flush=True)

models = {
    'Ridge': RidgeCV(alphas=np.logspace(-2, 3, 25), cv=5),
    'Lasso': LassoCV(alphas=np.logspace(-4, 0, 25), cv=5, max_iter=8000, random_state=42),
    'ElasticNet': ElasticNetCV(alphas=np.logspace(-4, 0, 25), l1_ratio=[0.1, 0.5, 0.7, 0.9],
                              cv=5, max_iter=8000, random_state=42)
}

val_metrics = {}
val_preds = {}
test_preds = {}
coef_dict = {}

for name, model in models.items():
    model.fit(X_train_proc, y_train_log)
    
    # Validation 예측 (log 복원: expm1)
    val_pred_log = model.predict(X_val_proc)
    val_pred = np.expm1(val_pred_log)
    val_actual = np.expm1(y_val_log)
    
    rmse = root_mean_squared_error(val_actual, val_pred)
    mae = mean_absolute_error(val_actual, val_pred)
    r2 = r2_score(val_actual, val_pred)
    zero_coefs = np.sum(model.coef_ == 0)
    
    val_metrics[name] = {
        'Alpha': model.alpha_,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Zero_Coefs': zero_coefs,
        'Active_Coefs': len(model.coef_) - zero_coefs
    }
    val_preds[name] = val_pred
    coef_dict[name] = model.coef_
    
    # Test 데이터 예측 및 저장
    test_pred_log = model.predict(X_test_proc)
    test_pred = np.expm1(test_pred_log)
    test_preds[name] = test_pred
    
    sub_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_pred})
    sub_output = f"data/submission_{name.lower()}.csv"
    sub_df.to_csv(sub_output, index=False)
    
    print(f"[{name}] 최적 Alpha={model.alpha_:.4f} | 검증 RMSE: ${rmse:,.0f} | R²: {r2:.4f} | 제거된 특성(0): {zero_coefs}/{len(model.coef_)}")

# -------------------------------------------------------------------
# 5. Sample Submission 기준 비교 평가
# -------------------------------------------------------------------
print(">> [4/5] Sample Submission(기준치)과 테스트 예측값 비교 분석...", flush=True)
sub_actual = sample_sub['SalePrice']
for name, pred in test_preds.items():
    diff_mae = mean_absolute_error(sub_actual, pred)
    print(f" - {name} vs Sample Submission 평균 편차(MAE): ${diff_mae:,.0f}")

# -------------------------------------------------------------------
# 6. 회귀계수 및 모델 성능 종합 시각화
# -------------------------------------------------------------------
print(">> [5/5] 회귀계수 및 성능 비교 시각화 차트 생성 중...", flush=True)
fig = plt.figure(figsize=(18, 12))

# (1) 제거된 변수 vs 살아남은 변수 수 비교
ax1 = plt.subplot(2, 2, 1)
model_names = list(models.keys())
zeros = [val_metrics[m]['Zero_Coefs'] for m in model_names]
actives = [val_metrics[m]['Active_Coefs'] for m in model_names]

x_idx = np.arange(len(model_names))
w = 0.4
ax1.bar(x_idx - w/2, zeros, width=w, label='제거된 특성(계수=0)', color='salmon')
ax1.bar(x_idx + w/2, actives, width=w, label='유지된 특성(계수!=0)', color='royalblue')
ax1.set_xticks(x_idx)
ax1.set_xticklabels(model_names, fontsize=12, fontweight='bold')
ax1.set_ylabel('특성(Feature) 개수', fontsize=12)
ax1.set_title('규제 모델별 특성 선택(Feature Selection) 비교', fontsize=14, pad=10)
ax1.legend(fontsize=11)
ax1.grid(axis='y', linestyle=':', alpha=0.7)

for i in x_idx:
    ax1.text(i - w/2, zeros[i] + 3, f"{zeros[i]}개", ha='center', fontsize=10)
    ax1.text(i + w/2, actives[i] + 3, f"{actives[i]}개", ha='center', fontsize=10)

# (2) 모델별 검증 R² 스코어 및 RMSE 비교
ax2 = plt.subplot(2, 2, 2)
r2_vals = [val_metrics[m]['R2'] for m in model_names]
rmse_vals = [val_metrics[m]['RMSE'] for m in model_names]

ax2_twin = ax2.twinx()
bars = ax2.bar(x_idx - w/2, r2_vals, width=w, color='teal', label='결정계수 (R² Score)')
lines = ax2_twin.plot(x_idx + w/2, rmse_vals, color='darkorange', marker='o', linewidth=2.5, markersize=8, label='RMSE ($)')

ax2.set_xticks(x_idx)
ax2.set_xticklabels(model_names, fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=12, color='teal')
ax2_twin.set_ylabel('RMSE ($)', fontsize=12, color='darkorange')
ax2.set_ylim(0.85, 0.95)
ax2.set_title('검증 데이터(Validation) 예측 성능 비교', fontsize=14, pad=10)
ax2.grid(axis='y', linestyle=':', alpha=0.7)

for i in x_idx:
    ax2.text(i - w/2, r2_vals[i] + 0.002, f"{r2_vals[i]:.4f}", ha='center', fontsize=10, color='teal', fontweight='bold')
    ax2_twin.text(i + w/2, rmse_vals[i] + 300, f"${rmse_vals[i]:,.0f}", ha='center', fontsize=10, color='darkorange', fontweight='bold')

# (3) 회귀계수 크기 분포 (Sorted Coefficients Plot)
ax3 = plt.subplot(2, 2, 3)
for name, color in zip(['Ridge', 'ElasticNet', 'Lasso'], ['royalblue', 'green', 'crimson']):
    sorted_coef = np.sort(coef_dict[name])
    ax3.plot(sorted_coef, label=name, color=color, linewidth=2)

ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('특성 정렬 인덱스', fontsize=12)
ax3.set_ylabel('정규화된 회귀계수(Weight) 크기', fontsize=12)
ax3.set_title('301개 회귀계수 크기 정렬 분포 (Sparsity 시각화)', fontsize=14, pad=10)
ax3.legend(fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)

# (4) Lasso 기준 가장 영향력이 큰 Top 15 특성 (양/음)
ax4 = plt.subplot(2, 2, 4)
lasso_coef = coef_dict['Lasso']
top_pos_idx = np.argsort(lasso_coef)[-10:]
top_neg_idx = np.argsort(lasso_coef)[:5]
top_idx = np.concatenate([top_neg_idx, top_pos_idx])

top_features = feature_names[top_idx]
top_weights = lasso_coef[top_idx]
colors = ['red' if w < 0 else 'blue' for w in top_weights]

y_pos = np.arange(len(top_features))
ax4.barh(y_pos, top_weights, color=colors, alpha=0.8)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(top_features, fontsize=9)
ax4.set_xlabel('Lasso 회귀 계수 (Log 집값 영향도)', fontsize=11)
ax4.set_title('Lasso 기준 집값에 가장 큰 영향을 미치는 핵심 특성 Top 15', fontsize=14, pad=10)
ax4.grid(axis='x', linestyle=':', alpha=0.6)

plt.tight_layout()
output_img = "house_price_regularization_comparison.png"
plt.savefig(output_img, dpi=300)
print(f">> 완료! 종합 시각화 결과가 '{output_img}' 파일로 저장되었습니다.", flush=True)
