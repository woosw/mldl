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

# -------------------------------------------------------------------
# 1. 한글 폰트 설정 (Windows 환경)
# -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------------
# 2. 데이터 로드 및 탐색
# -------------------------------------------------------------------
print(">> [1/5] Wine 데이터셋 로드...", flush=True)
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
target_names = [f"Class_{i} ({name})" for i, name in enumerate(wine.target_names)]
feature_names = wine.feature_names

print(f" - 샘플 수: {X.shape[0]}개, 특성 수: {X.shape[1]}개")
print(f" - 클래스 분포:\n{y.value_counts().sort_index()}")

# 스케일 차이 확인 (왜 StandardScaler가 필수인가?)
print("\n[특성별 스케일 범위 예시]")
print(f" - 'proline' 특성 범위: {X['proline'].min():.1f} ~ {X['proline'].max():.1f} (평균: {X['proline'].mean():.1f})")
print(f" - 'nonflavanoid_phenols' 특성 범위: {X['nonflavanoid_phenols'].min():.2f} ~ {X['nonflavanoid_phenols'].max():.2f} (평균: {X['nonflavanoid_phenols'].mean():.2f})")
print(f"   => 두 특성 간 약 {X['proline'].mean() / X['nonflavanoid_phenols'].mean():.0f}배 이상의 극심한 단위 차이 발생!")

# -------------------------------------------------------------------
# 3. 데이터 분할 (클래스 비율 유지 Stratified)
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -------------------------------------------------------------------
# 4. StandardScaler + 다중 클래스 로지스틱 회귀 파이프라인 구축
# -------------------------------------------------------------------
print("\n>> [2/5] 다중 클래스(Softmax) 로지스틱 회귀 모델 학습...", flush=True)

# 다중 클래스 교차 엔트로피(Cross-Entropy / Multinomial Loss) 기반 학습
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

model_pipeline.fit(X_train, y_train)

# -------------------------------------------------------------------
# 5. 모델 평가 (정확도, 분류 리포트, 소프트맥스 손실값)
# -------------------------------------------------------------------
print(">> [3/5] 테스트 데이터 평가...", flush=True)
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)

print(f"\n★ 모델 테스트 정확도(Accuracy): {acc * 100:.2f}%")
print(f"★ 다중 클래스 교차 엔트로피 손실(Log-Loss): {loss:.4f}\n")
print("[상세 분류 리포트 (Classification Report)]")
print(classification_report(y_test, y_pred, target_names=target_names))

# -------------------------------------------------------------------
# 6. 종합 시각화 (4분면 대시보드)
# -------------------------------------------------------------------
print(">> [4/5] 시각화 대시보드 생성 중...", flush=True)
fig = plt.figure(figsize=(18, 12))

# (1) 스케일링 전 vs 후 비교 (StandardScaler의 필요성 입증)
ax1 = plt.subplot(2, 2, 1)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

# 대표적으로 스케일 차이가 큰 4개 특성 선택
sample_features = ['alcohol', 'magnesium', 'proline', 'nonflavanoid_phenols']
X_sub_raw = X[sample_features]
X_sub_scaled = X_scaled[sample_features]

bplot1 = ax1.boxplot(
    [X_sub_raw[col] for col in sample_features],
    positions=np.arange(len(sample_features))*2.0 - 0.35,
    widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightcoral')
)
bplot2 = ax1.boxplot(
    [X_sub_scaled[col] for col in sample_features],
    positions=np.arange(len(sample_features))*2.0 + 0.35,
    widths=0.5, patch_artist=True, boxprops=dict(facecolor='cornflowerblue')
)
ax1.set_xticks(np.arange(len(sample_features))*2.0)
ax1.set_xticklabels(sample_features, fontsize=10, rotation=15)
ax1.set_yscale('symlog')  # 극심한 스케일 차이를 표현하기 위해 symlog 축 사용
ax1.set_ylabel('특성값 크기 (대칭 로그 스케일)', fontsize=11)
ax1.set_title('(1) StandardScaler 전/후 특성 분포 비교 (스케일 정규화)', fontsize=13, pad=10)
ax1.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['스케일링 전 (Raw)', '표준화 후 (StandardScaled)'], loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# (2) 혼동 행렬 (Confusion Matrix Heatmap)
ax2 = plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class_0', 'Class_1', 'Class_2'])
disp.plot(cmap='Blues', ax=ax2, colorbar=False)
ax2.set_title(f'(2) 혼동 행렬 (Confusion Matrix) - 정확도: {acc*100:.1f}%', fontsize=13, pad=10)
ax2.set_xlabel('예측 품종 (Predicted Class)', fontsize=11)
ax2.set_ylabel('실제 품종 (True Class)', fontsize=11)

# (3) 소프트맥스 예측 확률 분포 (Sample Softmax Probabilities)
ax3 = plt.subplot(2, 2, 3)
sample_indices = np.arange(min(15, len(y_test)))
bottom_vals = np.zeros(len(sample_indices))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for class_idx in range(3):
    probs = y_prob[sample_indices, class_idx]
    ax3.bar(
        sample_indices, probs, bottom=bottom_vals,
        label=f'P({target_names[class_idx]})', color=colors[class_idx], width=0.65, alpha=0.85
    )
    bottom_vals += probs

ax3.set_xlabel('테스트 샘플 인덱스', fontsize=11)
ax3.set_ylabel('소프트맥스 예측 확률 (합 = 1.0)', fontsize=11)
ax3.set_title('(3) 테스트 샘플별 소프트맥스(Softmax) 클래스 예측 확률', fontsize=13, pad=10)
ax3.set_xticks(sample_indices)
ax3.set_xticklabels([f"#{i}\n(정답:{y_test.iloc[i]})" for i in sample_indices], fontsize=9)
ax3.set_ylim(0, 1.1)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(axis='y', linestyle=':', alpha=0.6)

# (4) 2D PCA 투영 평면에서의 로지스틱 회귀 결정 경계면 (Decision Boundary)
ax4 = plt.subplot(2, 2, 4)
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(StandardScaler().fit_transform(X_train))
X_test_pca = pca.transform(StandardScaler().fit_transform(X_test))

# 2차원 투영 공간에서 로지스틱 회귀 재학습 (결정 경계 시각화 목적)
clf_2d = LogisticRegression(random_state=42)
clf_2d.fit(X_train_pca, y_train)

# 메쉬그리드 생성
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 결정 경계면 등고선 채우기
ax4.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.coolwarm)

# 실제 데이터 점 표시
scatter_colors = ['navy', 'darkgreen', 'firebrick']
for i, c_name in enumerate(['Class_0', 'Class_1', 'Class_2']):
    idx = (y_test == i)
    ax4.scatter(
        X_test_pca[idx, 0], X_test_pca[idx, 1],
        c=scatter_colors[i], label=f'실제 {c_name}', edgecolors='k', s=60
    )

ax4.set_xlabel(f'주성분 1 (설명분산비: {pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax4.set_ylabel(f'주성분 2 (설명분산비: {pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax4.set_title('(4) PCA 2D 투영 평면에서의 로지스틱 회귀 다중 결정 경계', fontsize=13, pad=10)
ax4.legend(loc='lower left', fontsize=10)
ax4.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
output_img = "wine_logistic_regression_result.png"
plt.savefig(output_img, dpi=300)
print(f">> 완료! 종합 시각화 결과가 '{output_img}' 파일로 저장되었습니다.", flush=True)
