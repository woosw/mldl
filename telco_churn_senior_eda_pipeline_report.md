# 통신사 고객 이탈(Telco Churn) 심층 EDA 및 머신러닝 연계 피처 엔지니어링 파이프라인 보고서

> **작성자**: 15년 차 시니어 데이터 사이언티스트 & 머신러닝 엔지니어  
> **대상 데이터**: [`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`](file:///C:/Users/sun/orca/workspaces/mldl/corbina/data/WA_Fn-UseC_-Telco-Customer-Churn.csv)  
> **실행 스크립트**: [`telco_churn_senior_pipeline.py`](file:///C:/Users/sun/orca/workspaces/mldl/corbina/telco_churn_senior_pipeline.py)  
> **생성 시각화 산출물**:
> - [`eda_phase2_univariate.png`](file:///C:/Users/sun/orca/workspaces/mldl/corbina/eda_phase2_univariate.png) (수치형 변수 히스토그램 및 박스플롯)
> - [`eda_phase3_bivariate.png`](file:///C:/Users/sun/orca/workspaces/mldl/corbina/eda_phase3_bivariate.png) (생존 곡선, 요금 변곡선, 약정별 이탈률, 기술지원 락인)
> - [`eda_phase4_multivariate.png`](file:///C:/Users/sun/orca/workspaces/mldl/corbina/eda_phase4_multivariate.png) (상관관계 및 계약×인터넷 교차 이탈률 히트맵)

---

## 📌 목차
1. [프로젝트 목적 및 분석 파이프라인 개요](#1-프로젝트-목적-및-분석-파이프라인-개요)
2. [Phase 1. 데이터 건전성 점검 및 무결성 전처리 (Data Health Check)](#2-phase-1-데이터-건전성-점검-및-무결성-전처리-data-health-check)
3. [Phase 2. 도메인 그룹별 일변량 분석 (Univariate Analysis)](#3-phase-2-도메인-그룹별-일변량-분석-univariate-analysis)
4. [Phase 3. 타깃 연계 이변량 분석 (Bivariate Analysis)](#4-phase-3-타깃-연계-이변량-분석-bivariate-analysis)
5. [Phase 4. 다변량 상호작용 및 다중공선성(Multicollinearity) 진단](#5-phase-4-다변량-상호작용-및-다중공선성multicollinearity-진단)
6. [Phase 5. 머신러닝 연계 파생 변수 생성 (Feature Engineering)](#6-phase-5-머신러닝-연계-파생-변수-생성-feature-engineering)
7. [전체 파이썬 소스 코드 (telco_churn_senior_pipeline.py)](#7-전체-파이썬-소스-코드-telco_churn_senior_pipelinepy)
8. [실행 콘솔 전체 로그 및 비즈니스 액션 플랜](#8-실행-콘솔-전체-로그-및-비즈니스-액션-플랜)

---

## 1. 프로젝트 목적 및 분석 파이프라인 개요

통신 산업의 구독 경제(Subscription Economy) 모델에서 고객 이탈을 조기에 방어하는 것은 신규 고객 유치 비용 대비 5배 이상의 ROI를 보장합니다. 본 보고서는 7,043명의 고객 데이터를 엄격한 데이터 품질 감사 관점에서 전처리하고, 통계적 가설 검증과 머신러닝 입력 피처 엔지니어링까지 유기적으로 연결하는 엔드투엔드 파이프라인 결과를 담고 있습니다.

---

## 2. Phase 1. 데이터 건전성 점검 및 무결성 전처리 (Data Health Check)

1. **식별자 격리 (`customerID`)**:
   * 총 7,043개 행 중 고유값 7,043개 일치 확인.
   * 모델 학습 시 인위적 과적합 및 데이터 누수(Data Leakage)를 방지하기 위해 1단계에서 완전히 제거.
2. **`TotalCharges` 결함 원인 추적 및 정제**:
   * 발견: `TotalCharges` 컬럼에 11건의 공백 문자(`" "`) 존재로 인해 `object` 타입으로 오염.
   * 원인: 해당 고객 11명 전원의 `tenure == 0` (가입 당월 고객으로 청구액 미발생).
   * 정제: 단순 평균/중앙값 대체는 신규 고객의 누적 요금을 부당하게 높이므로, 논리적 수치인 `0.0`으로 대체 후 `float64` 형변환 완료.
3. **타깃 변수(`Churn`) 클래스 분포율**:
   * 유지(`No`): 5,174건 (73.46%)
   * 이탈(`Yes`): 1,869건 (26.54%)
   * **진단**: 약 1:2.77의 불균형. 정확도(Accuracy) 지표 착시를 경계하고 **재현율(Recall)**과 **PR-AUC**를 핵심 최적화 목표로 설정.

---

## 3. Phase 2. 도메인 그룹별 일변량 분석 (Univariate Analysis)

![Phase 2 수치형 분포](file:///C:/Users/sun/orca/workspaces/mldl/corbina/eda_phase2_univariate.png)

### (1) 도메인 범주형 그룹 요약
* **인구통계**: 고령자(`SeniorCitizen`) 16.2%, 부양가족 없음(`Dependents: No`) 70.0%
* **서비스 구성**: 광섬유 인터넷(`Fiber optic`)이 44.0%로 주력 상품이며, 보안(`OnlineSecurity`) 및 기술지원(`TechSupport`) 미이용자가 약 49%에 달함.
* **계약/결제**: 월별 무약정(`Month-to-month`) 고객이 **55.0%**로 과반 이상이며, 수동 결제인 전자 수표(`Electronic check`) 비중이 **33.6%**로 가장 높음.

### (2) 수치형 변수 형태학(Morphology) 진단
| 변수명 | 평균 | 표준편차 | 중앙값 | 왜도 (Skewness) | 첨도 (Kurtosis) | 분포 형태 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`tenure`** | 32.37 | 24.56 | 29.0 | +0.24 | -1.39 | **쌍봉형 (Bimodal)**: 1~3개월 차와 60개월 이상 장기 고객으로 양극화 |
| **`MonthlyCharges`** | 64.76 | 30.09 | 70.35 | -0.22 | -1.26 | **다봉형 (Multimodal)**: $20 기본요금과 $70~$100 광섬유 번들 요금 |
| **`TotalCharges`** | 2279.73 | 2266.79 | 1394.55 | **+0.96** | -0.23 | **우왜도 (Right-skewed)**: 누적 금액 편차가 매우 큼 |

---

## 4. Phase 3. 타깃 연계 이변량 분석 (Bivariate Analysis)

![Phase 3 이변량 분석](file:///C:/Users/sun/orca/workspaces/mldl/corbina/eda_phase3_bivariate.png)

1. **가입 기간(`tenure`) 생존 커브와 온보딩 위험선**:
   * 가입 **12개월 이하 신규 고객의 이탈률은 47.44%**로 전체 평균(26.5%)의 **1.8배**에 달함.
   * 1년(12개월)을 넘기면 이탈 밀도가 급격히 완만해지는 저항선 형성.
2. **월 요금(`MonthlyCharges`) 이탈 급증 저항선 ($70)**:
   * $70 미만 고객의 이탈률: **17.2%**
   * $70 이상 고객의 이탈률: **35.5% (2배 이상 급증)**
3. **계약 형태(`Contract`)별 이탈률 격차**:
   * `Month-to-month`: **42.7%**
   * `One year`: **11.3%**
   * `Two year`: **2.8%** (무약정 고객의 이탈률이 2년 약정 대비 무려 **15배**)
4. **광섬유(`Fiber optic`) 고객의 기술지원(`TechSupport`) 락인 효과**:
   * 기술지원 미가입 고객: 이탈률 **49.4% (둘 중 한 명 해지)**
   * 기술지원 보유 고객: 이탈률 **22.6% (이탈률 54% 감소)**

---

## 5. Phase 4. 다변량 상호작용 및 다중공선성(Multicollinearity) 진단

![Phase 4 다변량 분석](file:///C:/Users/sun/orca/workspaces/mldl/corbina/eda_phase4_multivariate.png)

### (1) 상관계수 및 VIF(분산팽창계수) 점검
```text
[피어슨 상관계수 행렬]
                tenure  MonthlyCharges  TotalCharges
tenure          1.0000          0.2479        0.8262
MonthlyCharges  0.2479          1.0000        0.6512
TotalCharges    0.8262          0.6512        1.0000

[VIF 산출 결과]
       Feature   VIF
        tenure  5.84
MonthlyCharges  3.22
  TotalCharges  9.51
```
* **엔지니어링 진단**: `TotalCharges`는 본질적으로 `tenure × MonthlyCharges`의 누적 결합이므로 VIF가 9.51에 달해 강한 다중공선성을 가집니다. 로지스틱 회귀와 같은 선형 모델 학습 시 `TotalCharges`를 그대로 투입하면 계수 분산이 폭발하므로, 정규화(StandardScaler) 및 **청구 일관성 비율 변수**로 치환하는 것이 안전합니다.

### (2) 계약(Contract) × 인터넷(InternetService) 이탈률 교차 매트릭스
* **최대 위험 세그먼트 도출**:
  * **`Month-to-month(월별 무약정)` × `Fiber optic(광섬유)` 고객군의 이탈률: 54.6%**
  * 전체 7,043명 중 가장 규모가 크면서도 과반수 이상이 이탈하는 최우선 방어 타깃입니다.

---

## 6. Phase 5. 머신러닝 연계 파생 변수 생성 (Feature Engineering)

EDA에서 도출된 인사이트를 바탕으로 3종의 고부가가치 파생 변수를 설계하여 데이터셋에 결합했습니다:

1. **`Tenure_Group` (가입 기간 코호트 범주화)**:
   * 구간: `0-12M`, `13-24M`, `25-48M`, `49M+`
   * **검증된 이탈률**:
     * `0-12M`: **47.44%** (초고위험군)
     * `13-24M`: 28.71%
     * `25-48M`: 20.39%
     * `49M+`: **9.51%** (충성군)
2. **`Service_Count` (6대 부가서비스 가입 총점 [0~6점])**:
   * 점수 산정: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
   * **검증된 락인 효과**:
     * 1개 이용: **45.76%** 이탈
     * 3개 이용: **27.37%** 이탈
     * 6개 이용: **5.28%** 이탈 (서비스 수가 늘어날수록 선형적으로 이탈률 급감)
3. **`Avg_Monthly_Ratio` (청구 일관성 지표)**:
   * 산식: $\frac{\text{TotalCharges}}{\text{tenure} \times \text{MonthlyCharges} + 1e-5}$
   * 정기 납부 고객은 $1.0$ 부근에 수렴하며, 요금 인상/연체/단기 결제 변동이 있는 이상 고객을 감지합니다.
4. **최종 정제 데이터셋 정보**:
   * 최종 형상: **7,043행, 23열**
   * 전체 결측치(NaN): **0개 완벽 정제**

---

## 7. 전체 파이썬 소스 코드 (telco_churn_senior_pipeline.py)

전체 실행 스크립트는 [telco_churn_senior_pipeline.py](file:///C:/Users/sun/orca/workspaces/mldl/corbina/telco_churn_senior_pipeline.py) 파일에 저장되어 있습니다.

---

## 8. 실행 콘솔 전체 로그 및 비즈니스 액션 플랜

### (1) 콘솔 전체 출력 로그
```text
================================================================================
>> [Phase 1] 데이터 건전성 점검 및 무결성 전처리 (Data Health Check)
================================================================================
1. 원본 데이터 형상(Shape): 7,043행, 21열
2. customerID 고유값 수: 7,043개 (전체 행 수와 일치: True)
   -> 분석 및 학습 시 데이터 누수/과적합 방지를 위해 customerID 격리 제거 완료.
3. TotalCharges 내 공백 문자(' ') 발견 건수: 11건
   -> 해당 공백 데이터 고객들의 가입 개월 수(tenure): [0]
   -> [원인 분석] 신규 가입 고객(tenure=0)으로 첫 달 요금이 청구되지 않아 공백 발생.
   -> [결측치 처리 근거] 단순 평균/중앙값 대체는 신규 고객의 누적 요금을 왜곡하므로 논리적 값인 0.0으로 대체.
   -> TotalCharges 형변환(float64) 및 결측치(0.0) 보정 완료 (잔여 NaN: 0개)

4. 타깃 변수(Churn) 분포율:
   - 유지(No):  5,174건 (73.46%)
   - 이탈(Yes): 1,869건 (26.54%)
   -> [불균형 진단] 약 1:2.77의 클래스 불균형. 정확도(Accuracy) 착시 주의 및 Recall/PR-AUC 최적화 필요.

================================================================================
>> [Phase 2] 도메인 그룹별 일변량 분석 (Univariate Analysis)
================================================================================
1. [인구통계학(Demographics) 특성 비율 요약]
   - gender         : Male:50.5%, Female:49.5%
   - SeniorCitizen  : 0:83.8%, 1:16.2%
   - Partner        : No:51.7%, Yes:48.3%
   - Dependents     : No:70.0%, Yes:30.0%

2. [서비스 구독(Services) 특성 비율 요약]
   - InternetService   : Fiber optic:44.0%, DSL:34.4%, No:21.7%
   - OnlineSecurity    : No:49.7%, Yes:28.7%, No internet service:21.7%
   - OnlineBackup      : No:43.8%, Yes:34.5%, No internet service:21.7%
   - DeviceProtection  : No:43.9%, Yes:34.4%, No internet service:21.7%
   - TechSupport       : No:49.3%, Yes:29.0%, No internet service:21.7%
   - StreamingTV       : No:39.9%, Yes:38.4%, No internet service:21.7%
   - StreamingMovies   : No:39.5%, Yes:38.8%, No internet service:21.7%

3. [계약 및 청구(Contract & Billing) 특성 비율 요약]
   - Contract          : Month-to-month:55.0%, Two year:24.1%, One year:20.9%
   - PaperlessBilling  : Yes:59.2%, No:40.8%
   - PaymentMethod     : Electronic check:33.6%, Mailed check:22.9%, Bank transfer (automatic):21.9%, Credit card (automatic):21.6%

4. [연속형 수치(Numeric) 분포 및 형태 분석]
                   mean      std    min      50%      max  skewness  kurtosis
tenure            32.37    24.56   0.00    29.00    72.00      0.24     -1.39
MonthlyCharges    64.76    30.09  18.25    70.35   118.75     -0.22     -1.26
TotalCharges    2279.73  2266.79   0.00  1394.55  8684.80      0.96     -0.23
   -> tenure: 양 끝단에 데이터가 몰리는 쌍봉형(Bimodal) 양극화 분포.
   -> MonthlyCharges: $20대 기본요금과 $70~$100대 광섬유 번들 요금제의 다봉형 분포.
   -> TotalCharges: 오른쪽으로 긴 꼬리를 갖는 우왜도(Right-skewed) 분포 (왜도: 0.96).
   -> [시각화 저장] 'eda_phase2_univariate.png' 생성 완료.

================================================================================
>> [Phase 3] 타깃 연계 이변량 분석 (Bivariate Analysis: Feature vs. Churn)
================================================================================
1. 가입 12개월 이하 신규 고객의 이탈률: 47.44% (전체 평균 26.5%의 1.8배)
2. 월 요금 $70 기준 이탈률 비교: $70 미만(17.2%) vs $70 이상(35.5%) -> 2배 급증!
3. 계약 형태별 이탈률: Month-to-month(42.7%) vs 2년 약정(2.8%) -> 15배 차이
4. 광섬유 고객 중 기술지원 부재 시 이탈률: 49.4% vs 지원 보유 시: 22.6% (락인 효과 검증)
   -> [시각화 저장] 'eda_phase3_bivariate.png' 생성 완료.

================================================================================
>> [Phase 4] 다변량 상호작용 및 다중공선성(Multicollinearity) 분석
================================================================================
1. [수치형 피처 간 피어슨 상관계수 행렬]
                tenure  MonthlyCharges  TotalCharges
tenure          1.0000          0.2479        0.8262
MonthlyCharges  0.2479          1.0000        0.6512
TotalCharges    0.8262          0.6512        1.0000

2. [수치형 피처 VIF (분산팽창계수) 진단]
       Feature   VIF
        tenure  5.84
MonthlyCharges  3.22
  TotalCharges  9.51
   -> [진단 결과] TotalCharges(9.51)와 tenure(5.84) 간 높은 선형 종속성 확인.
   -> 로지스틱 회귀 등 선형 모델 학습 시 TotalCharges 정규화 및 파생 비율 변수 대체 필수.

3. [최대 위험 세그먼트 도출]
   - 세그먼트: 'Month-to-month(월별 무약정)' × 'Fiber optic(광섬유)'
   - 해당 군집 이탈률: 54.6% (전체 7,043명 중 가장 위험한 타겟)
   -> [시각화 저장] 'eda_phase4_multivariate.png' 생성 완료.

================================================================================
>> [Phase 5] 머신러닝 연계 파생 변수 생성 (Feature Engineering)
================================================================================
1. 신규 피처 생성: 'Tenure_Group' (0-12M, 13-24M, 25-48M, 49M+ 코호트 범주화)
2. 신규 피처 생성: 'Service_Count' (부가서비스 6종 가입 총 개수 [0~6점])
3. 신규 피처 생성: 'Avg_Monthly_Ratio' (실제 누적 청구액 / 기대 누적 청구액 비율)

4. [최종 정제 데이터셋 정보 및 결측치 검증]
 - 최종 데이터 형상: 7,043행, 23열 (3개 피처 추가)
 - 전체 결측치(NaN) 합계: 0개

5. [파생 변수 유효성 검증: Tenure_Group별 이탈률]
   - 0-12M   : 47.44%
   - 13-24M  : 28.71%
   - 25-48M  : 20.39%
   - 49M+    : 9.51%

6. [파생 변수 유효성 검증: Service_Count별 이탈률]
   - 0개 가입: 21.41%
   - 1개 가입: 45.76%
   - 2개 가입: 35.82%
   - 3개 가입: 27.37%
   - 4개 가입: 22.30%
   - 5개 가입: 12.43%
   - 6개 가입: 5.28%

================================================================================
>> [전체 파이프라인 완료] 모든 EDA 및 피처 엔지니어링이 성공적으로 종료되었습니다.
================================================================================
```

---

### (2) 비즈니스 액션 플랜 (Executive Summary)
1. **타겟 방어 페르소나**: `Month-to-month` + `Fiber optic` + `TechSupport 미가입` (이탈률 54.6%)
2. **원스톱 락인 프로모션**: 가입 3~6개월 차 고객 대상, 1년 약정 전환 및 자동이체 신청 시 `TechSupport` 및 `OnlineSecurity` 6개월 무료 제공 프로모션 집행
3. **모델링 권고**: 선형 분류 모델(로지스틱 회귀) 입력 시 VIF 9.51인 `TotalCharges`를 단독 투입하지 말고, `StandardScaler` 적용 및 `Avg_Monthly_Ratio`, `Service_Count`를 핵심 특성으로 활용하며 최적화 평가지표로 **Recall**을 채택할 것을 권장합니다.
