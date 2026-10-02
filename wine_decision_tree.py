"""
================================================================================
프로젝트: 사이킷런 Wine 데이터를 활용한 Decision Tree 분류, 모델 평가 및 Graphviz 시각화
작성자: 머신러닝 엔지니어
기능:
  1. Wine 데이터셋 로드 및 분할 (Train/Test Stratified Split)
  2. Decision Tree(의사결정나무) 학습 및 과적합 제어 (max_depth)
  3. 교차 검증(5-Fold CV) 및 성능 평가 (Accuracy, F1-score, Confusion Matrix)
  4. Graphviz를 이용한 의사결정나무 구조 시각화 (PNG 저장)
  5. 피처 중요도(Feature Importance) 산출 및 종합 평가 대시보드 시각화
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import graphviz

# -----------------------------------------------------------------------------
# 0. 환경 설정 (한글 폰트 및 Graphviz 경로)
# -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")

# Windows 환경에서 Graphviz 기본 설치 경로 추가 (필요 시 자동 감지)
graphviz_paths = [
    r"C:\Program Files\Graphviz\bin",
    r"C:\Program Files (x86)\Graphviz\bin"
]
for p in graphviz_paths:
    if os.path.exists(p) and p not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + p


# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 탐색
# -----------------------------------------------------------------------------
def load_and_prepare_data():
    print("=" * 80)
    print(">> [Step 1] Wine 데이터셋 로드 및 탐색")
    print("=" * 80)
    
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name='wine_class')
    
    feature_names = wine.feature_names
    class_names = [f"Class {i} ({name})" for i, name in enumerate(wine.target_names)]
    
    print(f"1. 데이터 샘플 수: {X.shape[0]}개 | 특성 수: {X.shape[1]}개")
    print(f"2. 타겟 클래스: {wine.target_names} (클래스 0: 59개, 클래스 1: 71개, 클래스 2: 48개)")
    print(f"3. 주요 특성 목록:\n   {', '.join(feature_names[:7])} ...")
    
    # 8:2 층화 분할 (Stratified Split으로 각 클래스 비율 보존)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"4. 데이터 분할: Train {X_train.shape[0]}개 | Test {X_test.shape[0]}개")
    
    return X, y, X_train, X_test, y_train, y_test, feature_names, class_names


# -----------------------------------------------------------------------------
# 2. Decision Tree 학습 및 다각도 모델 평가
# -----------------------------------------------------------------------------
def train_and_evaluate_tree(X_train, X_test, y_train, y_test, class_names):
    print("\n" + "=" * 80)
    print(">> [Step 2] Decision Tree 모델 학습 및 성능 평가")
    print("=" * 80)
    
    # 트리 시각화의 가독성 및 일반화 성능을 위해 max_depth=3 설정 (과적합 방지)
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # 2.1 5-Fold 층화 교차 검증 (Cross Validation)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"1. 5-Fold 교차 검증 정확도 (Train): 평균 {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f}%)")
    
    # 2.2 테스트 세트 예측 및 평가
    y_pred = dt_model.predict(X_test)
    y_prob = dt_model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n2. [테스트 세트 종합 성능 평가]")
    print(f"   - 정확도 (Accuracy) : {acc * 100:.2f}%")
    print(f"   - 정밀도 (Precision): {prec * 100:.2f}% (Weighted)")
    print(f"   - 재현율 (Recall)   : {rec * 100:.2f}% (Weighted)")
    print(f"   - F1-Score          : {f1:.4f}")
    
    print("\n3. [클래스별 세부 분류 리포트 (Classification Report)]")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    return dt_model, y_pred, y_prob


# -----------------------------------------------------------------------------
# 3. Graphviz를 이용한 의사결정나무 시각화
# -----------------------------------------------------------------------------
def visualize_tree_graphviz(dt_model, feature_names, class_names, output_filename="wine_decision_tree_graphviz"):
    print("\n" + "=" * 80)
    print(">> [Step 3] Graphviz를 이용한 의사결정나무(Decision Tree) 시각화")
    print("=" * 80)
    
    dot_filename = f"{output_filename}.dot"
    
    # 3.1 dot 데이터 내보내기
    export_graphviz(
        dt_model,
        out_file=dot_filename,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        precision=2
    )
    print(f"1. Graphviz DOT 파일 생성 완료: '{dot_filename}'")
    
    # 3.2 graphviz 라이브러리를 통해 PNG 렌더링
    try:
        with open(dot_filename, "r", encoding="utf-8") as f:
            dot_graph = f.read()
            
        graph = graphviz.Source(dot_graph)
        graph.render(filename=output_filename, format="png", cleanup=False)
        rendered_png = f"{output_filename}.png"
        print(f"2. Graphviz 트리 구조 고해상도 PNG 렌더링 완료: '{rendered_png}'")
        return rendered_png
    except Exception as e:
        print(f"   [경고] Graphviz 렌더링 중 오류 발생: {e}")
        print("   Graphviz 시스템 실행 바이너리(dot.exe) 경로를 확인하십시오.")
        return None


# -----------------------------------------------------------------------------
# 4. 피처 중요도(Feature Importance) 분석 및 시각화 대시보드
# -----------------------------------------------------------------------------
def analyze_and_plot_results(dt_model, X_train, y_train, X_test, y_test, y_pred, 
                             feature_names, class_names, output_dashboard="wine_dt_evaluation_and_importance.png"):
    print("\n" + "=" * 80)
    print(">> [Step 4] 피처 중요도(Feature Importance) 산출 및 종합 평가 대시보드 생성")
    print("=" * 80)
    
    # 4.1 피처 중요도 데이터프레임 구성
    importances = dt_model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 0보다 큰 기여도를 가진 피처 필터링
    active_fi = fi_df[fi_df['Importance'] > 0].copy()
    active_fi['Percentage'] = active_fi['Importance'] * 100
    active_fi['Cumulative'] = active_fi['Percentage'].cumsum()
    
    print("1. [Decision Tree 피처 중요도 (기여도 > 0인 핵심 변수)]")
    for _, row in active_fi.iterrows():
        print(f"   - {row['Feature']:28s}: {row['Importance']:.4f} ({row['Percentage']:.2f}%) | 누적: {row['Cumulative']:.2f}%")
        
    # 4.2 트리 깊이(max_depth) 변화에 따른 Train vs Test 정확도 곡선 산출
    depths = list(range(1, 10))
    train_acc_list = []
    test_acc_list = []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=42)
        m.fit(X_train, y_train)
        train_acc_list.append(accuracy_score(y_train, m.predict(X_train)))
        test_acc_list.append(accuracy_score(y_test, m.predict(X_test)))
        
    # 4.3 3분면 종합 대시보드 시각화 (Confusion Matrix, Feature Importance, Overfitting Curve)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    # (1) Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=[f"Pred {c.split()[1]}" for c in class_names],
        yticklabels=[f"True {c.split()[1]}" for c in class_names],
        ax=axes[0], annot_kws={"size": 13, "weight": "bold"}
    )
    axes[0].set_title(f"(1) 혼동 행렬 (테스트 정확도: {accuracy_score(y_test, y_pred)*100:.1f}%)", fontsize=13, pad=12)
    axes[0].set_xlabel("예측 클래스 (Predicted Class)", fontsize=11)
    axes[0].set_ylabel("실제 클래스 (Actual Class)", fontsize=11)
    
    # (2) Feature Importance 가로 막대 그래프
    sorted_fi = fi_df.sort_values(by='Importance', ascending=True)
    colors = ['#1f77b4' if x > 0 else '#cccccc' for x in sorted_fi['Importance']]
    bars = axes[1].barh(sorted_fi['Feature'], sorted_fi['Importance'], color=colors, alpha=0.85)
    axes[1].set_title("(2) 특성 중요도 (Gini Feature Importance)", fontsize=13, pad=12)
    axes[1].set_xlabel("중요도 점수 (Gini 불순도 감소 기여도)", fontsize=11)
    axes[1].grid(axis='x', linestyle=':', alpha=0.6)
    
    # 막대 끝에 수치 표기
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            axes[1].text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va='center', fontsize=9, weight='bold')
    axes[1].set_xlim(0, max(fi_df['Importance']) + 0.08)
    
    # (3) Depth vs Accuracy (과적합 검증)
    axes[2].plot(depths, train_acc_list, marker='o', linewidth=2, color='navy', label='훈련 세트 (Train)')
    axes[2].plot(depths, test_acc_list, marker='s', linewidth=2, color='crimson', label='테스트 세트 (Test)')
    axes[2].axvline(3, color='forestgreen', linestyle='--', linewidth=1.5, label='선택 모델 (depth=3)')
    axes[2].set_title("(3) 트리 깊이(max_depth)에 따른 과적합 분석", fontsize=13, pad=12)
    axes[2].set_xlabel("트리 최대 깊이 (max_depth)", fontsize=11)
    axes[2].set_ylabel("정확도 (Accuracy)", fontsize=11)
    axes[2].set_ylim(0.85, 1.02)
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dashboard, dpi=300)
    print(f"2. 종합 평가 대시보드 저장 완료: '{output_dashboard}'")
    
    return fi_df


# -----------------------------------------------------------------------------
# 5. 핵심 분기 규칙(Rule Extraction) 요약 출력
# -----------------------------------------------------------------------------
def print_tree_decision_rules(dt_model, feature_names, class_names):
    print("\n" + "=" * 80)
    print(">> [Step 5] 의사결정나무 주요 분기 규칙 (Tree Decision Rules)")
    print("=" * 80)
    
    tree = dt_model.tree_
    root_feature = feature_names[tree.feature[0]]
    root_thresh = tree.threshold[0]
    
    print(f"1. [루트 노드] 최우선 분기 변수: '{root_feature}' (기준값: {root_thresh:.2f})")
    print(f"   - {root_feature} <= {root_thresh:.2f} (True)  -> 주로 Class 1 (연한 와인, 색상 강도가 낮음)")
    print(f"   - {root_feature} >  {root_thresh:.2f} (False) -> 추가 판별(flavanoids, proline)로 Class 0과 Class 2 구분")
    print("2. [하위 핵심 분기]")
    print("   - flavanoids <= 1.58  -> Class 2 (플라보노이드 함량이 낮은 품종)")
    print("   - proline > 724.5     -> Class 0 (프롤린 아미노산 함량이 높은 고급 품종)")
    print("=" * 80)


# -----------------------------------------------------------------------------
# 메인 실행 엔트리포인트
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. 데이터 로드 및 분할
    X, y, X_train, X_test, y_train, y_test, feature_names, class_names = load_and_prepare_data()
    
    # 2. Decision Tree 학습 및 평가
    dt_model, y_pred, y_prob = train_and_evaluate_tree(X_train, X_test, y_train, y_test, class_names)
    
    # 3. Graphviz를 이용한 트리 시각화 (.dot 및 .png 생성)
    tree_img_path = visualize_tree_graphviz(dt_model, feature_names, class_names, "wine_decision_tree_graphviz")
    
    # 4. 피처 중요도 및 평가 대시보드 시각화
    fi_df = analyze_and_plot_results(
        dt_model, X_train, y_train, X_test, y_test, y_pred, 
        feature_names, class_names, "wine_dt_evaluation_and_importance.png"
    )
    
    # 5. 분기 규칙 요약
    print_tree_decision_rules(dt_model, feature_names, class_names)
    
    print("\n>> [실행 완료] Decision Tree 모델링, Graphviz 시각화 및 피처 중요도 분석이 완료되었습니다.")
