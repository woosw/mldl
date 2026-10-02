"""
================================================================================
프로젝트: Telco Churn EDA 기반 전처리 및 로지스틱 회귀(Logistic Regression) 모델링 & 평가
작성자: 15년 차 시니어 데이터 사이언티스트 & 머신러닝 엔지니어
기반 보고서: telco_churn_senior_eda_pipeline_report.md
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, fbeta_score, accuracy_score, precision_score, recall_score
)

# -------------------------------------------------------------------
# 0. 한글 폰트 및 스타일 설정 (Windows 환경)
# -------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")


# -------------------------------------------------------------------
# 1. EDA 인사이트 기반 전처리 및 피처 엔지니어링 파이프라인
# -------------------------------------------------------------------
def load_and_preprocess_data(filepath: str):
    print("=" * 80)
    print(">> [Step 1] EDA 보고서 기반 데이터 무결성 전처리 및 피처 엔지니어링")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"1. 원본 데이터 로드: {df.shape[0]:,}행, {df.shape[1]}열")
    
    # 1.1 식별자 제거 (데이터 누수 방지)
    df = df.drop(columns=['customerID'])
    print("2. customerID 식별자 격리 완료.")
    
    # 1.2 TotalCharges 공백(' ') 처리: tenure=0 신규 고객이므로 0.0원 대체
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan)).fillna(0.0)
    print("3. TotalCharges 공백 11건 -> 0.0원 대체 및 float64 변환 완료.")
    
    # 1.3 파생 변수 3종 생성 (EDA Phase 5 반영)
    # ① Tenure_Group: 코호트 범주화
    df['Tenure_Group'] = pd.cut(
        df['tenure'], bins=[-1, 12, 24, 48, 100], 
        labels=['0-12M', '13-24M', '25-48M', '49M+']
    ).astype(str)
    
    # ② Service_Count: 6대 부가서비스 가입 합산 점수 (0~6점)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Service_Count'] = (df[service_cols] == 'Yes').sum(axis=1)
    
    # ③ Avg_Monthly_Ratio: 누적 청구액 대비 기대 청구액 일관성 지표
    expected_charges = df['tenure'] * df['MonthlyCharges'] + 1e-5
    df['Avg_Monthly_Ratio'] = (df['TotalCharges'] / expected_charges).round(4)
    print("4. 파생 변수 3종(Tenure_Group, Service_Count, Avg_Monthly_Ratio) 생성 완료.")
    
    # 1.4 타겟 이진화
    y = (df['Churn'] == 'Yes').astype(int)
    X = df.drop(columns=['Churn'])
    
    # 1.5 컬럼 유형 분류
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Service_Count', 'Avg_Monthly_Ratio']
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    print(f"5. 피처 분류: 수치형 {len(num_cols)}개, 범주형 {len(cat_cols)}개")
    return X, y, num_cols, cat_cols


# -------------------------------------------------------------------
# 2. 로지스틱 회귀 모델 학습 (기본 vs 불균형 보정 vs 임계값 튜닝)
# -------------------------------------------------------------------
def train_and_evaluate_models(X, y, num_cols, cat_cols):
    print("\n" + "=" * 80)
    print(">> [Step 2] 로지스틱 회귀(Logistic Regression) 학습 및 평가 비교")
    print("=" * 80)
    
    # 2.1 층화 분할 (Stratified Split 8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"1. 데이터 분할: Train {X_train.shape[0]:,}개 | Test {X_test.shape[0]:,}개 (이탈률 26.5% 보존)")
    
    # 2.2 전처리 파이프라인 (수치형 정규화 + 범주형 원핫인코딩)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # 모델 1: 기본 로지스틱 회귀 (가중치 미적용, 임계값 0.5)
    pipe_default = Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_default.fit(X_train, y_train)
    
    # 모델 2: 불균형 보정 로지스틱 회귀 (class_weight='balanced')
    pipe_balanced = Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    pipe_balanced.fit(X_train, y_train)
    
    # 2.3 테스트 세트 예측 및 평가
    y_pred_def = pipe_default.predict(X_test)
    y_prob_def = pipe_default.predict_proba(X_test)[:, 1]
    
    y_pred_bal = pipe_balanced.predict(X_test)
    y_prob_bal = pipe_balanced.predict_proba(X_test)[:, 1]
    
    # 모델 3: 기본 모델 확률에서 재현율 극대화를 위한 최적 임계값 적용 (Threshold = 0.30)
    best_thresh = 0.30
    y_pred_thresh = (y_prob_def >= best_thresh).astype(int)
    
    # 2.4 성능 비교표 출력
    def get_metrics(y_true, y_pred, y_prob):
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall(재현율)': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'F2-Score(Recall중시)': fbeta_score(y_true, y_pred, beta=2),
            'ROC-AUC': roc_auc_score(y_true, y_prob)
        }
    
    metrics_df = pd.DataFrame({
        'Model 1 (기본: Thr=0.5)': get_metrics(y_test, y_pred_def, y_prob_def),
        'Model 2 (불균형 보정 Balanced)': get_metrics(y_test, y_pred_bal, y_prob_bal),
        'Model 3 (임계값 튜닝 Thr=0.3)': get_metrics(y_test, y_pred_thresh, y_prob_def)
    }).T
    
    print("\n[3개 모델 성능 비교 요약]")
    print(metrics_df.round(4).to_string())
    
    print("\n[Model 2 (불균형 보정 Balanced) 상세 분류 리포트]")
    print(classification_report(y_test, y_pred_bal, target_names=['유지(No)', '이탈(Yes)']))
    
    return pipe_default, pipe_balanced, X_test, y_test, y_prob_def, y_prob_bal, y_pred_def, y_pred_bal


# -------------------------------------------------------------------
# 3. 회귀계수 및 오즈비(Odds Ratio) 통계적 해석
# -------------------------------------------------------------------
def interpret_odds_ratios(pipeline, cat_cols, num_cols):
    print("\n" + "=" * 80)
    print(">> [Step 3] 로지스틱 회귀계수(Beta) 및 오즈비(Odds Ratio = exp(Beta)) 해석")
    print("=" * 80)
    
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['prep']
    
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_cat_names)
    
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    
    or_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient(Beta)': coefs,
        'Odds_Ratio(exp(Beta))': odds_ratios
    }).sort_values(by='Odds_Ratio(exp(Beta))', ascending=False)
    
    print("1. [이탈 확률을 가장 크게 증가시키는 요인 Top 5 (오즈비 > 1)]")
    top_churn = or_df.head(5)
    for _, r in top_churn.iterrows():
        print(f"   - {r['Feature']:35s}: Beta={r['Coefficient(Beta)']:+.4f} | 오즈비={r['Odds_Ratio(exp(Beta))']:.2f}배 증가")
        
    print("\n2. [이탈 확률을 가장 크게 방어하는 락인 요인 Top 5 (오즈비 < 1)]")
    top_stay = or_df.tail(5).iloc[::-1]
    for _, r in top_stay.iterrows():
        print(f"   - {r['Feature']:35s}: Beta={r['Coefficient(Beta)']:+.4f} | 오즈비={r['Odds_Ratio(exp(Beta))']:.2f}배 (이탈위험 대폭 감소)")
        
    return or_df


# -------------------------------------------------------------------
# 4. 종합 평가 대시보드 시각화 (4분면)
# -------------------------------------------------------------------
def visualize_results(y_test, y_pred_def, y_pred_bal, y_prob_def, y_prob_bal, or_df):
    print("\n" + "=" * 80)
    print(">> [Step 4] 모델 평가 종합 대시보드 시각화 생성 중...")
    print("=" * 80)
    
    fig = plt.figure(figsize=(18, 12))
    
    # (1) 혼동 행렬 비교: 기본(0.5) vs 불균형 보정(Balanced)
    ax1 = plt.subplot(2, 2, 1)
    cm_bal = confusion_matrix(y_test, y_pred_bal)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_bal, display_labels=['유지(No)', '이탈(Yes)'])
    disp.plot(cmap='Blues', ax=ax1, colorbar=False)
    rec = recall_score(y_test, y_pred_bal) * 100
    ax1.set_title(f'(1) 불균형 보정 모델 혼동 행렬 (재현율: {rec:.1f}%)', fontsize=13, pad=10)
    ax1.set_xlabel('예측 라벨')
    ax1.set_ylabel('실제 라벨')
    
    # (2) ROC 곡선 및 PR(Precision-Recall) 곡선
    ax2 = plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_prob_bal)
    auc_score = roc_auc_score(y_test, y_prob_bal)
    ax2.plot(fpr, tpr, color='darkorange', linewidth=2.5, label=f'Balanced 모델 (AUC = {auc_score:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', linestyle='--', label='무작위 기준선 (AUC = 0.5)')
    ax2.set_title(f'(2) ROC 곡선 (AUC = {auc_score:.3f})', fontsize=13, pad=10)
    ax2.set_xlabel('위양성률 (1 - 특이도)')
    ax2.set_ylabel('진양성률 (재현율, Recall)')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # (3) 분류 임계값(Threshold) 변화에 따른 Precision vs Recall 트레이드오프
    ax3 = plt.subplot(2, 2, 3)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_def)
    
    # F1 및 F2 스코어 계산
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    f2_scores = (1 + 2**2) * (precisions * recalls) / (4 * precisions + recalls + 1e-10)
    
    ax3.plot(thresholds, precisions[:-1], 'b--', label='정밀도 (Precision)', linewidth=2)
    ax3.plot(thresholds, recalls[:-1], 'g-', label='재현율 (Recall)', linewidth=2.5)
    ax3.plot(thresholds, f2_scores[:-1], 'm-.', label='F2-Score (이탈 방어 최적화)', linewidth=2)
    
    # 최적 임계점 (F2 기준)
    best_idx = np.argmax(f2_scores)
    best_th = thresholds[best_idx]
    ax3.axvline(best_th, color='crimson', linestyle=':', linewidth=2, label=f'최적 임계점 ({best_th:.2f})')
    ax3.set_title(f'(3) 임계값 튜닝 곡선 (방어 최적 임계점 = {best_th:.2f})', fontsize=13, pad=10)
    ax3.set_xlabel('분류 확률 임계값 (Threshold)')
    ax3.set_ylabel('지표 값 (0.0 ~ 1.0)')
    ax3.legend(loc='lower center', fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # (4) 오즈비(Odds Ratio) 상위/하위 10개 핵심 피처
    ax4 = plt.subplot(2, 2, 4)
    top_pos = or_df.head(6)
    top_neg = or_df.tail(6)
    plot_or = pd.concat([top_neg, top_pos]).sort_values(by='Odds_Ratio(exp(Beta))')
    
    colors = ['#2b5c8f' if x < 1 else '#d95f02' for x in plot_or['Odds_Ratio(exp(Beta))']]
    bars = ax4.barh(plot_or['Feature'], plot_or['Odds_Ratio(exp(Beta))'], color=colors, alpha=0.85)
    ax4.axvline(1.0, color='black', linestyle='--', linewidth=1.2)
    ax4.set_title('(4) 핵심 특성별 오즈비 (Odds Ratio, 이탈 위험도 배수)', fontsize=13, pad=10)
    ax4.set_xlabel('오즈비 (1.0 기준: 1.0보다 크면 이탈 위험 증가, 작으면 유지)')
    for bar in bars:
        w = bar.get_width()
        ax4.text(w + 0.05, bar.get_y() + bar.get_height()/2, f"{w:.2f}배", va='center', fontsize=9)
    ax4.set_xlim(0, max(plot_or['Odds_Ratio(exp(Beta))']) + 0.6)
    ax4.grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    output_png = "telco_churn_logistic_regression_result.png"
    plt.savefig(output_png, dpi=300)
    print(f">> [완료] 시각화 대시보드가 '{output_png}' 파일로 성공적으로 저장되었습니다.")


# -------------------------------------------------------------------
# 메인 실행 엔트리포인트
# -------------------------------------------------------------------
if __name__ == '__main__':
    data_file = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    X, y, num_cols, cat_cols = load_and_preprocess_data(data_file)
    pipe_def, pipe_bal, X_test, y_test, y_prob_def, y_prob_bal, y_pred_def, y_pred_bal = train_and_evaluate_models(
        X, y, num_cols, cat_cols
    )
    or_df = interpret_odds_ratios(pipe_bal, cat_cols, num_cols)
    visualize_results(y_test, y_pred_def, y_pred_bal, y_prob_def, y_prob_bal, or_df)
    
    print("\n" + "=" * 80)
    print(">> [전체 머신러닝 파이프라인 종료] 로지스틱 회귀 학습 및 평가 완료.")
    print("=" * 80)
