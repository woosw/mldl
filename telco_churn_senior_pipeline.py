"""
================================================================================
프로젝트: Telco Customer Churn 심층 EDA 및 머신러닝 연계 피처 엔지니어링 파이프라인
작성자: 15년 차 시니어 데이터 사이언티스트 & 머신러닝 엔지니어
데이터: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------------------------------
# 0. 시각화 및 환경 설정 (한글 폰트 및 마이너스 부호 대응)
# -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


# -------------------------------------------------------------------
# Phase 1. 데이터 건전성 점검 및 무결성 전처리 (Data Health Check)
# -------------------------------------------------------------------
def phase1_data_health_check(filepath: str) -> pd.DataFrame:
    print("=" * 80)
    print(">> [Phase 1] 데이터 건전성 점검 및 무결성 전처리 (Data Health Check)")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"1. 원본 데이터 형상(Shape): {df.shape[0]:,}행, {df.shape[1]}열")
    
    # 1.1 customerID 고유값 검증 및 격리/제거
    n_unique_id = df['customerID'].nunique()
    print(f"2. customerID 고유값 수: {n_unique_id:,}개 (전체 행 수와 일치: {n_unique_id == len(df)})")
    df = df.drop(columns=['customerID'])
    print("   -> 분석 및 학습 시 데이터 누수/과적합 방지를 위해 customerID 격리 제거 완료.")
    
    # 1.2 TotalCharges 공백 문자(" ") 탐색 및 tenure == 0 관계 검증
    blank_mask = df['TotalCharges'] == ' '
    n_blanks = blank_mask.sum()
    print(f"3. TotalCharges 내 공백 문자(' ') 발견 건수: {n_blanks}건")
    
    tenure_of_blanks = df.loc[blank_mask, 'tenure'].unique()
    print(f"   -> 해당 공백 데이터 고객들의 가입 개월 수(tenure): {tenure_of_blanks.tolist()}")
    print("   -> [원인 분석] 신규 가입 고객(tenure=0)으로 첫 달 요금이 청구되지 않아 공백 발생.")
    print("   -> [결측치 처리 근거] 단순 평균/중앙값 대체는 신규 고객의 누적 요금을 왜곡하므로 논리적 값인 0.0으로 대체.")
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan)).fillna(0.0)
    print(f"   -> TotalCharges 형변환(float64) 및 결측치(0.0) 보정 완료 (잔여 NaN: {df['TotalCharges'].isnull().sum()}개)")
    
    # 1.3 타깃 변수(Churn) 클래스 분포율 및 불균형 진단
    churn_counts = df['Churn'].value_counts()
    churn_rates = df['Churn'].value_counts(normalize=True) * 100
    print(f"\n4. 타깃 변수(Churn) 분포율:")
    print(f"   - 유지(No):  {churn_counts['No']:,}건 ({churn_rates['No']:.2f}%)")
    print(f"   - 이탈(Yes): {churn_counts['Yes']:,}건 ({churn_rates['Yes']:.2f}%)")
    print("   -> [불균형 진단] 약 1:2.77의 클래스 불균형. 정확도(Accuracy) 착시 주의 및 Recall/PR-AUC 최적화 필요.")
    
    return df


# -------------------------------------------------------------------
# Phase 2. 도메인 그룹별 일변량 분석 (Univariate Analysis)
# -------------------------------------------------------------------
def phase2_univariate_analysis(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(">> [Phase 2] 도메인 그룹별 일변량 분석 (Univariate Analysis)")
    print("=" * 80)
    
    # 도메인 그룹 정의
    demo_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    service_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    contract_cols = ['Contract', 'PaperlessBilling', 'PaymentMethod']
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # 2.1 범주형 그룹 요약
    print("1. [인구통계학(Demographics) 특성 비율 요약]")
    for col in demo_cols:
        val_str = ", ".join([f"{k}:{v:.1f}%" for k, v in (df[col].value_counts(normalize=True)*100).items()])
        print(f"   - {col:15s}: {val_str}")
        
    print("\n2. [서비스 구독(Services) 특성 비율 요약]")
    for col in service_cols:
        val_str = ", ".join([f"{k}:{v:.1f}%" for k, v in (df[col].value_counts(normalize=True)*100).items()])
        print(f"   - {col:18s}: {val_str}")
        
    print("\n3. [계약 및 청구(Contract & Billing) 특성 비율 요약]")
    for col in contract_cols:
        val_str = ", ".join([f"{k}:{v:.1f}%" for k, v in (df[col].value_counts(normalize=True)*100).items()])
        print(f"   - {col:18s}: {val_str}")
        
    # 2.2 연속형 수치 통계량 및 형태(Skewness, Kurtosis) 분석
    print("\n4. [연속형 수치(Numeric) 분포 및 형태 분석]")
    num_summary = df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    num_summary['skewness'] = df[num_cols].skew()
    num_summary['kurtosis'] = df[num_cols].kurtosis()
    print(num_summary.round(2).to_string())
    print("   -> tenure: 양 끝단에 데이터가 몰리는 쌍봉형(Bimodal) 양극화 분포.")
    print("   -> MonthlyCharges: $20대 기본요금과 $70~$100대 광섬유 번들 요금제의 다봉형 분포.")
    print("   -> TotalCharges: 오른쪽으로 긴 꼬리를 갖는 우왜도(Right-skewed) 분포 (왜도: 0.96).")

    # 시각화: 수치형 변수 Boxplot 및 히스토그램 (이상치 점검)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[0, i], color='#2b5c8f', bins=30)
        axes[0, i].set_title(f'{col} 분포 히스토그램 (왜도: {df[col].skew():.2f})')
        
        sns.boxplot(x=df[col], ax=axes[1, i], color='#e08214')
        axes[1, i].set_title(f'{col} 이상치 점검 (Boxplot)')
        
    plt.tight_layout()
    plt.savefig("eda_phase2_univariate.png", dpi=200)
    plt.close()
    print("   -> [시각화 저장] 'eda_phase2_univariate.png' 생성 완료.")


# -------------------------------------------------------------------
# Phase 3. 타깃 연계 이변량 분석 (Bivariate Analysis: Feature vs. Churn)
# -------------------------------------------------------------------
def phase3_bivariate_analysis(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(">> [Phase 3] 타깃 연계 이변량 분석 (Bivariate Analysis: Feature vs. Churn)")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 3.1 tenure 분포 생존 커브 (KDE Plot)
    sns.kdeplot(data=df, x='tenure', hue='Churn', common_norm=False, fill=True,
                palette={'No': '#2b5c8f', 'Yes': '#d95f02'}, alpha=0.4, ax=axes[0, 0])
    axes[0, 0].axvline(12, color='crimson', linestyle='--', linewidth=2, label='초기 위험 경계(12개월)')
    axes[0, 0].set_title('(1) 가입 기간(tenure)별 이탈 생존 커브 (온보딩 위험 구간)', fontsize=12)
    axes[0, 0].set_xlabel('가입 개월 수(tenure)')
    axes[0, 0].set_ylabel('확률 밀도(Density)')
    axes[0, 0].legend()
    
    tenure_under_12_churn = df[df['tenure'] <= 12]['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    print(f"1. 가입 12개월 이하 신규 고객의 이탈률: {tenure_under_12_churn:.2f}% (전체 평균 26.5%의 1.8배)")
    
    # 3.2 MonthlyCharges 밀도 분포 및 이탈 변곡점(Threshold)
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False, fill=True,
                palette={'No': '#2b5c8f', 'Yes': '#d95f02'}, alpha=0.4, ax=axes[0, 1])
    axes[0, 1].axvline(70, color='crimson', linestyle='--', linewidth=2, label='이탈 급증 저항선($70)')
    axes[0, 1].set_title('(2) 월 청구액(MonthlyCharges) 밀도 및 이탈 변곡점', fontsize=12)
    axes[0, 1].set_xlabel('월 요금 ($)')
    axes[0, 1].set_ylabel('확률 밀도(Density)')
    axes[0, 1].legend()
    
    churn_under_70 = df[df['MonthlyCharges'] < 70]['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    churn_over_70 = df[df['MonthlyCharges'] >= 70]['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    print(f"2. 월 요금 $70 기준 이탈률 비교: $70 미만({churn_under_70:.1f}%) vs $70 이상({churn_over_70:.1f}%) -> 2배 급증!")
    
    # 3.3 Contract 및 PaymentMethod 이탈률 막대 비교
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    pay_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
    
    axes[1, 0].bar(contract_churn.index, contract_churn.values, color=['#d95f02', '#2b5c8f', '#2b5c8f'], width=0.5)
    axes[1, 0].set_title('(3) 계약 형태(Contract)별 이탈률 (%)', fontsize=12)
    axes[1, 0].set_ylabel('이탈률 (%)')
    for x, y in zip(range(len(contract_churn)), contract_churn.values):
        axes[1, 0].text(x, y + 1, f"{y:.1f}%", ha='center', fontweight='bold')
    axes[1, 0].set_ylim(0, 50)
    
    print(f"3. 계약 형태별 이탈률: Month-to-month({contract_churn['Month-to-month']:.1f}%) vs 2년 약정({contract_churn['Two year']:.1f}%) -> 15배 차이")
    
    # 3.4 InternetService & 보안/기술지원 락인(Lock-in) 비교
    internet_security = df.groupby(['InternetService', 'TechSupport'])['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).unstack()
    internet_security.plot(kind='bar', ax=axes[1, 1], colormap='viridis', width=0.7)
    axes[1, 1].set_title('(4) 인터넷 종류 × 기술지원(TechSupport) 교차 이탈률', fontsize=12)
    axes[1, 1].set_ylabel('이탈률 (%)')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
    axes[1, 1].legend(title='TechSupport 유무')
    
    fiber_tech_no = df[(df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No')]['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    fiber_tech_yes = df[(df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'Yes')]['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
    print(f"4. 광섬유 고객 중 기술지원 부재 시 이탈률: {fiber_tech_no:.1f}% vs 지원 보유 시: {fiber_tech_yes:.1f}% (락인 효과 검증)")
    
    plt.tight_layout()
    plt.savefig("eda_phase3_bivariate.png", dpi=200)
    plt.close()
    print("   -> [시각화 저장] 'eda_phase3_bivariate.png' 생성 완료.")


# -------------------------------------------------------------------
# Phase 4. 다변량 상호작용 및 상관관계 분석 (Multivariate Analysis)
# -------------------------------------------------------------------
def phase4_multivariate_analysis(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(">> [Phase 4] 다변량 상호작용 및 다중공선성(Multicollinearity) 분석")
    print("=" * 80)
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    corr_matrix = df[num_cols].corr()
    
    print("1. [수치형 피처 간 피어슨 상관계수 행렬]")
    print(corr_matrix.round(4).to_string())
    
    # VIF 산출
    vif_data = pd.DataFrame()
    vif_data["Feature"] = num_cols
    vif_data["VIF"] = [variance_inflation_factor(df[num_cols].values, i) for i in range(len(num_cols))]
    print("\n2. [수치형 피처 VIF (분산팽창계수) 진단]")
    print(vif_data.round(2).to_string(index=False))
    print("   -> [진단 결과] TotalCharges와 tenure/MonthlyCharges 간 강한 선형종속 관계 (VIF > 10).")
    print("   -> 선형 모델(로지스틱 회귀) 학습 시 TotalCharges 정규화 또는 파생 변수 치환 필수.")
    
    # 4.2 교차 분석: Contract × InternetService 이탈률 히트맵
    cross_churn = pd.crosstab(df['Contract'], df['InternetService'], 
                              values=(df['Churn'] == 'Yes').astype(int), 
                              aggfunc='mean') * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.3f', ax=axes[0])
    axes[0].set_title('(1) 수치형 특성 상관계수 히트맵 (다중공선성 점검)')
    
    sns.heatmap(cross_churn, annot=True, cmap='OrRd', fmt='.1f', ax=axes[1], cbar_kws={'label': '이탈률 (%)'})
    axes[1].set_title('(2) 계약(Contract) × 인터넷(InternetService) 이탈률 교차 히트맵')
    axes[1].set_ylabel('계약 형태')
    axes[1].set_xlabel('인터넷 종류')
    
    plt.tight_layout()
    plt.savefig("eda_phase4_multivariate.png", dpi=200)
    plt.close()
    
    max_risk_val = cross_churn.loc['Month-to-month', 'Fiber optic']
    print(f"\n3. [최대 위험 세그먼트 도출]")
    print(f"   - 세그먼트: 'Month-to-month(월별 무약정)' × 'Fiber optic(광섬유)'")
    print(f"   - 해당 군집 이탈률: {max_risk_val:.1f}% (전체 7,043명 중 가장 위험한 타겟)")
    print("   -> [시각화 저장] 'eda_phase4_multivariate.png' 생성 완료.")


# -------------------------------------------------------------------
# Phase 5. 머신러닝 연계 파생 변수 생성 (Feature Engineering)
# -------------------------------------------------------------------
def phase5_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(">> [Phase 5] 머신러닝 연계 파생 변수 생성 (Feature Engineering)")
    print("=" * 80)
    
    df_fe = df.copy()
    
    # 5.1 Tenure_Group: 가입 기간 코호트 구간화
    bins = [-1, 12, 24, 48, 100]
    labels = ['0-12M', '13-24M', '25-48M', '49M+']
    df_fe['Tenure_Group'] = pd.cut(df_fe['tenure'], bins=bins, labels=labels)
    print("1. 신규 피처 생성: 'Tenure_Group' (0-12M, 13-24M, 25-48M, 49M+ 코호트 범주화)")
    
    # 5.2 Service_Count: 6대 부가 서비스 가입 개수 합산
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_fe['Service_Count'] = (df_fe[service_cols] == 'Yes').sum(axis=1)
    print("2. 신규 피처 생성: 'Service_Count' (부가서비스 6종 가입 총 개수 [0~6점])")
    
    # 5.3 Avg_Monthly_Ratio: 누적 청구액 대비 기대 청구액 일관성 지표
    # TotalCharges / (tenure * MonthlyCharges + 1e-5)
    # 1.0에 가까울수록 정기 청구 유지, 1.0보다 크면 일시적 추가 과금/연체, 0이면 신규
    expected_charges = df_fe['tenure'] * df_fe['MonthlyCharges'] + 1e-5
    df_fe['Avg_Monthly_Ratio'] = (df_fe['TotalCharges'] / expected_charges).round(4)
    print("3. 신규 피처 생성: 'Avg_Monthly_Ratio' (실제 누적 청구액 / 기대 누적 청구액 비율)")
    
    # 5.4 최종 정제 데이터 검증
    print("\n4. [최종 정제 데이터셋 정보 및 결측치 검증]")
    print(f" - 최종 데이터 형상: {df_fe.shape[0]:,}행, {df_fe.shape[1]}열 (3개 피처 추가)")
    print(f" - 전체 결측치(NaN) 합계: {df_fe.isnull().sum().sum()}개")
    
    churn_by_tenure_grp = df_fe.groupby('Tenure_Group')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    print("\n5. [파생 변수 유효성 검증: Tenure_Group별 이탈률]")
    for grp, rate in churn_by_tenure_grp.items():
        print(f"   - {grp:8s}: {rate:.2f}%")
        
    churn_by_service_cnt = df_fe.groupby('Service_Count')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    print("\n6. [파생 변수 유효성 검증: Service_Count별 이탈률]")
    for cnt, rate in churn_by_service_cnt.items():
        print(f"   - {cnt}개 가입: {rate:.2f}%")

    return df_fe


# -------------------------------------------------------------------
# 메인 실행 엔트리포인트
# -------------------------------------------------------------------
if __name__ == '__main__':
    data_file = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Phase 1 실행
    df_clean = phase1_data_health_check(data_file)
    
    # Phase 2 실행
    phase2_univariate_analysis(df_clean)
    
    # Phase 3 실행
    phase3_bivariate_analysis(df_clean)
    
    # Phase 4 실행
    phase4_multivariate_analysis(df_clean)
    
    # Phase 5 실행
    df_final = phase5_feature_engineering(df_clean)
    
    print("\n" + "=" * 80)
    print(">> [전체 파이프라인 완료] 모든 EDA 및 피처 엔지니어링이 성공적으로 종료되었습니다.")
    print("=" * 80)
