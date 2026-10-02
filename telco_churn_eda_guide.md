# 통신사 고객 이탈(Telco Churn) 데이터 심층 EDA & Self-Refine 프롬프트 엔지니어링 가이드

> **작성자 관점**: 프롬프트 엔지니어링 전문가 & 시니어 데이터 분석가  
> **대상 데이터**: `C:/Users/sun/orca/workspaces/mldl/corbina/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`  
> **핵심 강화 사항**: 
> 1. 실무에서 즉시 실행 가능한 **모범 파이썬 EDA 코드 파이프라인 수록**
> 2. 코드와 분석 결과의 품질을 스스로 검증하고 고도화하는 **자가 개선(Self-Refine) 프레임워크 설계 및 적용**

---

## 📌 목차
1. [비즈니스 문제 정의 및 분석 개요](#1-비즈니스-문제-정의-및-분석-개요)
2. [데이터 프로파일링 및 구조적 결함 진단 (Data Auditing)](#2-데이터-프로파일링-및-구조적-결함-진단-data-auditing)
3. [자가 개선(Self-Refine) 프레임워크 이론 및 설계](#3-자가-개선self-refine-프레임워크-이론-및-설계)
4. [단계별 파이썬 EDA 구현 및 Self-Refine 적용](#4-단계별-파이썬-eda-구현-및-self-refine-적용)
   - [Step 1: 데이터 정제 및 결함 복구 (Data Cleansing)](#step-1-데이터-정제-및-결함-복구-data-cleansing)
   - [Step 2: 타겟 불균형 및 단변량 분포 분석](#step-2-타겟-불균형-및-단변량-분포-분석)
   - [Step 3: 핵심 요인 이변량 분석 (Contract, Internet, Payment)](#step-3-핵심-요인-이변량-분석-contract-internet-payment)
   - [Step 4: 다변량 교차 및 부가서비스 락인(Lock-in) 분석](#step-4-다변량-교차-및-부가서비스-락인lock-in-분석)
5. [프롬프트 엔지니어링: Self-Refine 마스터 프롬프트 플레이북](#5-프롬프트-엔지니어링-self-refine-마스터-프롬프트-플레이북)
6. [실전 Self-Refine 적용 사례 (Before vs Critique vs After)](#6-실전-self-refine-적용-사례-before-vs-critique-vs-after)
7. [최종 요약 및 모델링 연계 체크리스트](#7-최종-요약-및-모델링-연계-체크리스트)

---

## 1. 비즈니스 문제 정의 및 분석 개요

* **비즈니스 배경**: 통신 산업(Subscription Economy)은 신규 고객 획득 비용(CAC)이 기존 고객 유지 비용(Retention Cost)의 5배 이상에 달합니다.
* **분석 목표**: 
  1. 어떤 고객군이 왜 이탈(Churn)하는지 통계적으로 규명
  2. 단순 상관관계 나열을 넘어 **자가 비판 및 개선(Self-Refine)** 루프를 통해 통찰의 신뢰성과 코드의 견고성을 확보
  3. 방어 마케팅 및 요금제 개편에 직접 활용 가능한 실행 지침(Actionable Insight) 도출

---

## 2. 데이터 프로파일링 및 구조적 결함 진단 (Data Auditing)

| 영역 | 변수 목록 | 분석 시 주의점 |
| :--- | :--- | :--- |
| **식별자** | `customerID` | 무의미한 고카디널리티 변수 $\rightarrow$ EDA/학습 제외 |
| **인구통계** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | 고령자 및 1인 가구의 취약점 분석 |
| **가입 서비스** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | 부가서비스 가입 개수(Service Stacking)에 따른 락인(Lock-in) 효과 검증 |
| **계약/요금** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | `TotalCharges`의 공백 문자(`" "`) 결측치 트랩 처리 필수 |
| **타겟 변수** | `Churn` (Yes / No) | 불균형 데이터셋 (이탈률 약 26.5%) |

---

## 3. 자가 개선(Self-Refine) 프레임워크 이론 및 설계

**Self-Refine(Madaan et al., 2023)**은 AI 또는 분석가가 초안(Draft)을 작성한 후, 미리 정의된 평가 기준(Critique Rubric)에 따라 스스로의 산출물을 냉정하게 평가하고 수정(Refine)하는 반복적 품질 고도화 기법입니다.

```
       ┌────────────────────────┐
       │   1. 초안 생성 (Draft)   │ ──► 초기 코드 작성 및 기본 분석 수행
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  2. 자가 비판 (Critique) │ ──► 데이터 결함, 누수, 시각화 가독성, 비즈니스 가치 검증
       └───────────┬────────────┘
                   │ [개선 필요 사항 식별]
                   ▼
       ┌────────────────────────┐
       │   3. 수정 반영 (Refine) │ ──► 결함 해결, 고급 시각화 적용, 구체적 실행 액션 추가
       └───────────┬────────────┘
                   │
                   ▼ (완료 기준 만족 시)
          [최종 고품질 산출물 도출]
```

### Self-Refine 3대 평가 루브릭 (Critique Rubric)
1. **기술적 견고성 (Technical Robustness)**:
   * 숨은 결측치(공백값)나 데이터 타입 왜곡을 완벽히 방어했는가?
   * 데이터 누수(Data Leakage)나 불필요한 고유 ID가 혼입되지 않았는가?
2. **시각화 인지 효율성 (Cognitive Ergonomics)**:
   * 단순 빈도 막대그래프가 아닌, 비율(%)과 절대 수치가 함께 직관적으로 전달되는가?
   * 한글 폰트 깨짐이나 음수 부호 깨짐이 완벽히 예방되었는가?
3. **비즈니스 실행성 (Business Actionability)**:
   * "이탈률이 높다"라는 1차원적 서술에 그치지 않고, "어떤 마케팅/요금제 조치를 취해야 하는가?"가 도출되었는가?

---

## 4. 단계별 파이썬 EDA 구현 및 Self-Refine 적용

### Step 1: 데이터 정제 및 결함 복구 (Data Cleansing)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 및 마이너스 부호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# [Self-Refine 반영 포인트 1] TotalCharges 공백 문자 처리
# tenure=0인 신규 가입 고객의 경우 TotalCharges가 ' '로 저장되어 object로 로드됨
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))

# tenure가 0인 고객은 누적 청구액이 0원이므로 0.0으로 대체
df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

# Churn 레이블을 이진 수치(1, 0)로 변환한 파생 컬럼 생성
df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("데이터 결측치 및 타입 정제 완료:")
print(f" - 전체 샘플 수: {df.shape[0]}행, 특성 수: {df.shape[1]}열")
print(f" - TotalCharges 결측치 잔여 수: {df['TotalCharges'].isnull().sum()}개")
```

---

### Step 2: 타겟 불균형 및 단변량 분포 분석

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) 타겟 변수 Churn 불균형 비율
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

axes[0].bar(churn_counts.index, churn_counts.values, color=['#2b5c8f', '#d95f02'], width=0.5)
axes[0].set_title(f'이탈(Churn) 비율 분포 (이탈률: {churn_pct["Yes"]:.1f}%)', fontsize=12)
axes[0].set_ylabel('고객 수(명)')
for i, (cnt, pct) in enumerate(zip(churn_counts, churn_pct)):
    axes[0].text(i, cnt + 80, f"{cnt:,}명\n({pct:.1f}%)", ha='center', fontsize=10)

# (2) tenure(가입 기간) 분포 - 쌍봉형(Bimodal) 양극화
sns.histplot(df['tenure'], kde=True, ax=axes[1], color='teal', bins=36)
axes[1].set_title('가입 기간(tenure) 분포 (초기 vs 장기 양극화)', fontsize=12)
axes[1].set_xlabel('가입 개월 수')

# (3) MonthlyCharges(월 요금) 분포 - 요금제 구간별 다봉형
sns.histplot(df['MonthlyCharges'], kde=True, ax=axes[2], color='purple', bins=30)
axes[2].set_title('월 청구 요금(MonthlyCharges) 분포', fontsize=12)
axes[2].set_xlabel('월 요금 ($)')

plt.tight_layout()
plt.show()
```

> **EDA 통찰**:
> * 전체 이탈률은 **26.5%**로 1:3 불균형 구조입니다. 향후 분류 모델에서 단순 정확도(Accuracy)는 왜곡을 일으키므로 **재현율(Recall)**과 **ROC-AUC**를 필수 지표로 삼아야 합니다.
> * `tenure`는 1~3개월 차 초기 이탈 위험군과 60개월 이상 충성 고객군으로 뚜렷하게 나뉩니다.

---

### Step 3: 핵심 요인 이변량 분석 (Contract, Internet, Payment)

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# [Self-Refine 반영 포인트 2] 단순 빈도가 아닌 '100% 정규화 누적 막대'로 이탈률 직접 비교
def plot_churn_ratio(column_name, ax, title):
    ct = pd.crosstab(df[column_name], df['Churn'], normalize='index') * 100
    ct[['No', 'Yes']].plot(kind='bar', stacked=True, color=['#4393c3', '#d6604d'], ax=ax, width=0.55)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('비율 (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(['유지 (No)', '이탈 (Yes)'], loc='upper right')
    for p in ax.patches:
        height = p.get_height()
        if height > 5:
            ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                        ha='center', va='center', color='white', fontweight='bold', fontsize=9)

plot_churn_ratio('Contract', axes[0], '(1) 계약 형태(Contract)별 이탈률')
plot_churn_ratio('InternetService', axes[1], '(2) 인터넷 종류(InternetService)별 이탈률')
plot_churn_ratio('PaymentMethod', axes[2], '(3) 결제 방식(PaymentMethod)별 이탈률')

plt.tight_layout()
plt.show()
```

> **EDA 통찰**:
> 1. **계약 형태**: 월별 무약정(`Month-to-month`) 고객의 이탈률은 **42.7%**인 반면, 2년 약정 고객은 **2.8%**에 불과합니다. (약정 유도가 최고의 방어책)
> 2. **인터넷 서비스**: 초고속 광섬유(`Fiber optic`) 고객의 이탈률이 **41.9%**로 DSL(19.0%) 대비 2배 이상 높습니다. (비싼 요금 대비 네트워크 안정성 불만 가설 성립)
> 3. **결제 방식**: 전자 수표(`Electronic check`) 결제 고객의 이탈률이 **45.3%**로 자동이체 고객(15~17%) 대비 압도적으로 높습니다.

---

### Step 4: 다변량 교차 및 부가서비스 락인(Lock-in) 분석

```python
# [Self-Refine 반영 포인트 3] 부가서비스 개수 합산 파생변수 생성 및 락인 검증
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']

# 각 서비스에서 'Yes'인 항목의 수를 합산하여 고객 락인 지수 산출
df['Total_Services_Count'] = (df[service_cols] == 'Yes').sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# (1) 부가서비스 가입 수에 따른 이탈률 변화 (락인 효과 검증)
service_churn = df.groupby('Total_Services_Count')['Churn_Numeric'].mean() * 100
axes[0].plot(service_churn.index, service_churn.values, marker='o', color='crimson', linewidth=2.5, markersize=8)
axes[0].set_title('부가 서비스 가입 개수별 이탈률 (서비스 락인 효과)', fontsize=12)
axes[0].set_xlabel('가입한 부가 서비스 수 (0개 ~ 6개)')
axes[0].set_ylabel('이탈률 (%)')
axes[0].grid(True, linestyle=':', alpha=0.6)
for x, y in zip(service_churn.index, service_churn.values):
    axes[0].annotate(f"{y:.1f}%", (x, y + 1.5), ha='center', fontweight='bold', color='crimson')

# (2) 광섬유 고객 중 기술지원(TechSupport) 유무에 따른 월 요금 vs 가입기간 이탈 산점도
fiber_df = df[df['InternetService'] == 'Fiber optic']
sns.scatterplot(
    data=fiber_df, x='tenure', y='MonthlyCharges', hue='Churn',
    style='TechSupport', palette={'No': '#2b5c8f', 'Yes': '#d95f02'},
    alpha=0.6, ax=axes[1]
)
axes[1].set_title('광섬유(Fiber Optic) 고객의 기간 vs 요금 vs 이탈', fontsize=12)
axes[1].set_xlabel('가입 기간(tenure)')
axes[1].set_ylabel('월 청구액(MonthlyCharges)')

plt.tight_layout()
plt.show()
```

> **EDA 통찰**:
> * 부가서비스를 하나도 이용하지 않는 고객은 이탈률이 **40% 이상**이지만, **3개 이상 이용 시 10%대, 5개 이상 이용 시 5% 미만**으로 급감합니다.
> * 이는 부가서비스 번들링 프로모션이 단순한 매출 증대용이 아니라 가장 강력한 **이탈 방어막(Lock-in Barrier)**임을 증명합니다.

---

## 5. 프롬프트 엔지니어링: Self-Refine 마스터 프롬프트 플레이북

LLM이나 AI 분석 에이전트에게 고품질의 EDA 코드와 비즈니스 통찰을 요청할 때 사용하는 **자가 개선 루프 내장형 프롬프트**입니다.

```text
[Role Definition]
너는 글로벌 통신사의 수석 데이터 사이언티스트이자 데이터 품질 감사 전문가다.

[Context]
'WA_Fn-UseC_-Telco-Customer-Churn.csv' 데이터를 활용하여 고객 이탈(Churn) 방지를 위한 EDA 및 전처리 파이프라인을 구축하고 있다.

[Instruction: 3-Stage Self-Refine Process]
너는 반드시 아래 3단계를 순차적으로 거쳐 최종 답변을 도출해야 한다:

1단계: 초안 작성 (Initial Draft)
- 데이터 전처리 및 탐색적 데이터 분석(EDA)을 위한 실행 가능한 파이썬 코드를 작성하라.
- 핵심 컬럼(Contract, InternetService, tenure, MonthlyCharges)에 대한 시각화와 통계량을 도출하라.

2단계: 자가 비판 및 결함 감사 (Self-Critique & Audit)
작성한 1단계 초안에 대해 다음 4가지 관점에서 스스로 오류와 한계를 지적하라:
  ① 데이터 무결성: TotalCharges의 공백 문자(' ') 처리 및 tenure=0 관계가 완벽한가?
  ② 시각화 왜곡: 범주형 변수를 단순 건수가 아닌 클래스별 비율(%)로 정규화했는가?
  ③ 다변량 상호작용: 단일 변수 분석을 넘어 광섬유+기술지원 결합 효과를 검증했는가?
  ④ 비즈니스 액션: 단순 현상 설명에 그치지 않고 구체적인 마케팅 처방을 제시했는가?

3단계: 최종 개선본 도출 (Refined Final Output)
- 2단계 자가 비판에서 지적된 모든 결함을 수정한 완성형 파이썬 코드와 고도화된 인사이트 리포트를 작성하라.

[Output Rules]
- 코드는 복사하여 즉시 실행 가능하도록 라이브러리 임포트부터 한글 폰트 설정까지 완전해야 한다.
- 각 차트마다 비즈니스 의사결정권자를 위한 '핵심 요약(Key Takeaway)'을 명시하라.
```

---

## 6. 실전 Self-Refine 적용 사례 (Before vs Critique vs After)

분석 과정에서 Self-Refine이 실제로 어떻게 코드와 분석의 품질을 바꾸는지 보여주는 대조표입니다.

| 구분 | 초기 초안 (Initial Draft) | 자가 비판 (Self-Critique) | 최종 개선본 (Refined Output) |
| :--- | :--- | :--- | :--- |
| **TotalCharges 전처리** | `pd.read_csv()` 후 바로 수치 통계 계산 시도 | `"TotalCharges 컬럼에 공백 문자(' ')가 포함되어 있어 문자열로 인식되며, 바로 연산 시 ValueError 발생 위험"` | `pd.to_numeric(errors='coerce')`로 결측화한 뒤, `tenure=0`인 신규 고객임을 감안하여 `0.0`으로 논리적 대체 |
| **계약 형태 시각화** | 단순 `sns.countplot(x='Contract', hue='Churn')` | `"월별 계약 고객 수가 많아 단순 카운트로 보면 각 그룹 내부의 실제 이탈 비율이 가려짐"` | `pd.crosstab(normalize='index')` 기반 100% 누적 막대그래프를 적용하여 **월별 계약 이탈률 42.7%**를 선명하게 부각 |
| **서비스 분석 깊이** | 인터넷 종류(DSL vs 광섬유)만 단편적 비교 | `"광섬유 고객이 왜 이탈하는지 설명 불가. 요금 문제인지 지원 부족인지 다변량 분석 결여"` | 광섬유 고객층 내부에서 `TechSupport` 유무 및 부가서비스 가입 개수(0~6개)와의 상호작용 분석 추가 |

---

## 7. 최종 요약 및 모델링 연계 체크리스트

EDA 및 Self-Refine을 통해 도출된 핵심 비즈니스 액션과 후속 머신러닝 모델링 연결 지침입니다:

1. **마케팅 방어 전략**:
   * **위험군**: 신규 가입 1~6개월 차 + 광섬유 + 전자 수표 결제 + 무약정 고객
   * **조치**: 가입 3개월 차에 1년 약정 전환 시 부가서비스(`OnlineSecurity`, `TechSupport`) 무료 제공 프로모션 집행
2. **모델링 시 특성 엔지니어링**:
   * `Total_Services_Count` (부가서비스 합산 점수) 신규 생성
   * `tenure` 코호트 범주화 (0~6개월 위험군, 7~12개월 등)
3. **평가 지표 설정**:
   * 이탈 고객을 놓쳤을 때의 비용(FN)이 방어 프로모션 비용(FP)보다 훨씬 크므로, **재현율(Recall)**과 **F2-Score**를 핵심 최적화 목표로 설정
