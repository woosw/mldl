import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. 환경 설정 및 한글 폰트 지정 (Windows 환경)
# -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------------
# 2. 데이터 로드 및 데이터 무결성 정제 파이프라인
# -------------------------------------------------------------------
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# [감사 반영 1] TotalCharges의 공백 문자(' ') 11건 정제 및 실수형 변환
# tenure가 0인 신규 가입 고객이므로 누적 청구액 0.0원으로 논리적 결측 대체
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan)).fillna(0.0)

# [감사 반영 2] 불필요한 식별자 제거 및 수치형 타겟 라벨 생성
df_clean = df.drop(columns=['customerID']).copy()
df_clean['Churn_Binary'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

# [감사 반영 3] 서비스 락인(Lock-in) 지수 파생 변수 생성
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df_clean['Total_Services'] = (df_clean[service_cols] == 'Yes').sum(axis=1)

print(">> [데이터 정제 감사 완료]")
print(f" - 데이터셋 크기: {df_clean.shape[0]}행, {df_clean.shape[1]}열")
print(f" - TotalCharges 결측치 수: {df_clean['TotalCharges'].isnull().sum()}개 (Dtype: {df_clean['TotalCharges'].dtype})")
print(f" - 전체 이탈률: {df_clean['Churn_Binary'].mean() * 100:.2f}%\n")

# -------------------------------------------------------------------
# 3. 고도화된 EDA 시각화 대시보드 (6분면)
# -------------------------------------------------------------------
fig = plt.figure(figsize=(20, 12))

# (1) 계약 형태(Contract)별 100% 정규화 이탈 비율
ax1 = plt.subplot(2, 3, 1)
contract_churn = pd.crosstab(df_clean['Contract'], df_clean['Churn'], normalize='index') * 100
contract_churn[['No', 'Yes']].plot(kind='bar', stacked=True, color=['#2b5c8f', '#d95f02'], ax=ax1, width=0.55)
ax1.set_title('(1) 계약 형태별 실제 이탈률 (100% 누적)', fontsize=12, pad=10)
ax1.set_ylabel('비율 (%)', fontsize=11)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.legend(['유지(No)', '이탈(Yes)'], loc='upper right')
for p in ax1.patches:
    h = p.get_height()
    if h > 5:
        ax1.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., p.get_y() + h / 2.),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=10)

# (2) 인터넷 종류(InternetService)별 100% 정규화 이탈 비율
ax2 = plt.subplot(2, 3, 2)
internet_churn = pd.crosstab(df_clean['InternetService'], df_clean['Churn'], normalize='index') * 100
internet_churn[['No', 'Yes']].plot(kind='bar', stacked=True, color=['#2b5c8f', '#d95f02'], ax=ax2, width=0.55)
ax2.set_title('(2) 인터넷 종류별 실제 이탈률 (100% 누적)', fontsize=12, pad=10)
ax2.set_ylabel('비율 (%)', fontsize=11)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.legend(['유지(No)', '이탈(Yes)'], loc='upper right')
for p in ax2.patches:
    h = p.get_height()
    if h > 5:
        ax2.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., p.get_y() + h / 2.),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=10)

# (3) 결제 수단(PaymentMethod)별 이탈률 비교
ax3 = plt.subplot(2, 3, 3)
payment_churn = (df_clean.groupby('PaymentMethod')['Churn_Binary'].mean() * 100).sort_values(ascending=False)
bars = ax3.barh(payment_churn.index, payment_churn.values, color=['#d95f02' if x > 30 else '#2b5c8f' for x in payment_churn.values])
ax3.set_title('(3) 결제 수단별 이탈률 (수동 vs 자동이체)', fontsize=12, pad=10)
ax3.set_xlabel('이탈률 (%)', fontsize=11)
for bar in bars:
    w = bar.get_width()
    ax3.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontweight='bold', fontsize=10)
ax3.set_xlim(0, 55)

# (4) 가입 기간(tenure)에 따른 이탈/유지 KDE 확률밀도 분포
ax4 = plt.subplot(2, 3, 4)
sns.kdeplot(data=df_clean, x='tenure', hue='Churn', common_norm=False, fill=True,
            palette={'No': '#2b5c8f', 'Yes': '#d95f02'}, alpha=0.4, ax=ax4)
ax4.set_title('(4) 가입 기간(tenure)별 이탈 밀도 곡선 (초기 위험 구간)', fontsize=12, pad=10)
ax4.set_xlabel('가입 개월 수', fontsize=11)
ax4.set_ylabel('확률 밀도(Density)', fontsize=11)
ax4.axvline(12, color='crimson', linestyle='--', linewidth=1.5, label='1년 위험 경계선')
ax4.legend(loc='upper right')

# (5) 다변량 교차: 광섬유(Fiber Optic) 고객의 TechSupport 유무별 이탈률
ax5 = plt.subplot(2, 3, 5)
fiber_users = df_clean[df_clean['InternetService'] == 'Fiber optic']
fiber_tech_churn = pd.crosstab(fiber_users['TechSupport'], fiber_users['Churn'], normalize='index') * 100
fiber_tech_churn[['No', 'Yes']].plot(kind='bar', stacked=True, color=['#2b5c8f', '#d95f02'], ax=ax5, width=0.5)
ax5.set_title('(5) 광섬유 고객의 기술지원(TechSupport) 유무별 이탈률', fontsize=12, pad=10)
ax5.set_ylabel('비율 (%)', fontsize=11)
ax5.set_xticklabels(['기술지원 없음(No)', '기술지원 이용(Yes)'], rotation=0)
ax5.legend(['유지(No)', '이탈(Yes)'], loc='upper right')
for p in ax5.patches:
    h = p.get_height()
    if h > 5:
        ax5.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., p.get_y() + h / 2.),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=10)

# (6) 서비스 가입 개수(Total Services)에 따른 락인(Lock-in) 곡선
ax6 = plt.subplot(2, 3, 6)
lockin_churn = (df_clean.groupby('Total_Services')['Churn_Binary'].mean() * 100)
ax6.plot(lockin_churn.index, lockin_churn.values, marker='o', color='purple', linewidth=2.5, markersize=8)
ax6.set_title('(6) 부가 서비스 가입 개수별 이탈률 (서비스 락인 효과)', fontsize=12, pad=10)
ax6.set_xlabel('가입한 부가 서비스 수 (0개 ~ 6개)', fontsize=11)
ax6.set_ylabel('이탈률 (%)', fontsize=11)
ax6.grid(True, linestyle=':', alpha=0.6)
for x, y in zip(lockin_churn.index, lockin_churn.values):
    ax6.annotate(f"{y:.1f}%", (x, y + 2), ha='center', fontweight='bold', color='purple', fontsize=10)
ax6.set_ylim(0, 50)

plt.tight_layout()
output_img = "telco_churn_refined_eda.png"
plt.savefig(output_img, dpi=300)
print(f">> [완료] 정제된 EDA 시각화 결과가 '{output_img}' 파일로 저장되었습니다.")
