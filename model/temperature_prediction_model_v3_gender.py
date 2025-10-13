import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_and_preprocess_data(file_path):
    """데이터 로드 및 전처리"""
    print("데이터 로딩 중...")
    df = pd.read_csv(file_path)
    print(f"원본 데이터 크기: {df.shape}")
    
    # s046의 temp_median이 0인 데이터 제거
    print("S046의 temp_median이 0인 데이터 제거 중...")
    df_clean = df[~((df['sid'] == 'S046') & (df['TEMP_median'] == 0))]
    print(f"정리된 데이터 크기: {df_clean.shape}")
    print(f"제거된 데이터 수: {df.shape[0] - df_clean.shape[0]}")
    
    return df_clean

def create_temperature_labels_v3(df):
    """TEMP_median 기준으로 온도 라벨 생성 (추움: 33도 미만, 더움: 35도 초과)"""
    print("온도 라벨 생성 중... (추움: 33도 미만, 더움: 35도 초과)")
    
    print(f"추움 기준: 33도 미만")
    print(f"더움 기준: 35도 초과")
    print(f"쾌적 기준: 33도 이상 35도 이하")
    
    # 라벨 생성: 0=추움, 1=쾌적, 2=더움
    def assign_label(temp):
        if temp < 33.0:
            return 0  # 추움
        elif temp > 35.0:
            return 2  # 더움
        else:
            return 1  # 쾌적
    
    df['temperature_label'] = df['TEMP_median'].apply(assign_label)
    
    # 라벨 분포 확인
    label_counts = df['temperature_label'].value_counts().sort_index()
    print("\n라벨 분포:")
    print(f"추움 (0): {label_counts[0]}개 ({label_counts[0]/len(df)*100:.1f}%)")
    print(f"쾌적 (1): {label_counts[1]}개 ({label_counts[1]/len(df)*100:.1f}%)")
    print(f"더움 (2): {label_counts[2]}개 ({label_counts[2]/len(df)*100:.1f}%)")
    
    return df

def prepare_features_v3_gender(df):
    """특성 준비 (5개 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, gender)"""
    print("특성 준비 중... (5개 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, gender)")
    
    # 기본 특성들
    basic_features = ['bmi', 'mean_sa02', 'HRV_SDNN', 'HR_mean']
    
    # Gender 원핫인코딩
    gender_dummies = pd.get_dummies(df['gender'], prefix='Gender')
    df_encoded = pd.concat([df, gender_dummies], axis=1)
    
    # 사용할 특성 선택
    feature_columns = basic_features + list(gender_dummies.columns)
    
    X = df_encoded[feature_columns]
    y = df_encoded['temperature_label']
    
    print(f"특성 수: {len(feature_columns)}")
    print(f"특성 목록: {feature_columns}")
    
    return X, y, feature_columns

def train_catboost_model(X, y, feature_columns):
    """CatBoost 모델 학습"""
    print("CatBoost 모델 학습 중...")
    
    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"학습 데이터 크기: {X_train.shape}")
    print(f"검증 데이터 크기: {X_val.shape}")
    
    # CatBoost 모델 설정 (과적합 방지)
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        eval_metric='Accuracy'
    )
    
    # 모델 학습
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        plot=False
    )
    
    return model, X_train, X_val, y_train, y_val

def evaluate_model(model, X_val, y_val, feature_columns):
    """모델 평가"""
    print("모델 평가 중...")
    
    # 예측
    y_pred = model.predict(X_val)
    
    # 정확도
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n검증 정확도: {accuracy:.4f}")
    
    # 분류 리포트
    print("\n분류 리포트:")
    print(classification_report(y_val, y_pred, 
                              target_names=['추움', '쾌적', '더움']))
    
    # 혼동 행렬
    cm = confusion_matrix(y_val, y_pred)
    print("\n혼동 행렬:")
    print(cm)
    
    # 특성 중요도
    feature_importance = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n특성 중요도:")
    print(feature_importance_df)
    
    return accuracy, feature_importance_df

def plot_results(model, X_val, y_val, feature_importance_df):
    """결과 시각화"""
    print("결과 시각화 중...")
    
    # 예측
    y_pred = model.predict(X_val)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['추움', '쾌적', '더움'],
                yticklabels=['추움', '쾌적', '더움'])
    plt.title('Confusion Matrix (V3 + Gender)')
    plt.ylabel('실제')
    plt.xlabel('예측')
    
    # 특성 중요도 시각화
    plt.subplot(1, 2, 2)
    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.title('Feature Importance (V3 + Gender)')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('temperature_prediction_results_v3_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_classification_analysis(model, X_val, y_val, feature_columns, feature_importance_df):
    """상세한 분류 분석"""
    print("상세 분류 분석 중...")
    
    # 예측
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    # 정확도 계산
    accuracy = accuracy_score(y_val, y_pred)
    
    # 클래스별 정확도
    class_names = ['추움', '쾌적', '더움']
    class_accuracies = []
    for i in range(3):
        class_mask = y_val == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_val[class_mask], y_pred[class_mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 혼동 행렬
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0,0])
    axes[0,0].set_title(f'혼동 행렬\n전체 정확도: {accuracy:.4f}')
    axes[0,0].set_ylabel('실제')
    axes[0,0].set_xlabel('예측')
    
    # 2. 클래스별 정확도
    bars = axes[0,1].bar(class_names, class_accuracies, 
                        color=['skyblue', 'lightgreen', 'salmon'])
    axes[0,1].set_ylabel('정확도')
    axes[0,1].set_title('클래스별 정확도')
    axes[0,1].set_ylim(0, 1)
    
    # 막대 위에 정확도 값 표시
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{acc:.3f}', ha='center', va='bottom')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 특성 중요도
    top_features = feature_importance_df.head(10)
    axes[1,0].barh(range(len(top_features)), top_features['importance'])
    axes[1,0].set_yticks(range(len(top_features)))
    axes[1,0].set_yticklabels(top_features['feature'])
    axes[1,0].set_xlabel('중요도')
    axes[1,0].set_title('특성 중요도 (상위 10개)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 예측 확률 분포
    for i, class_name in enumerate(class_names):
        class_proba = y_pred_proba[y_val == i, :]
        if len(class_proba) > 0:
            axes[1,1].hist(class_proba[:, i], bins=20, alpha=0.6, 
                          label=f'{class_name} (실제)', density=True)
    axes[1,1].set_xlabel('예측 확률')
    axes[1,1].set_ylabel('빈도')
    axes[1,1].set_title('클래스별 예측 확률 분포')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_analysis_v3_gender_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("=== 실시간 온도 예측 모델 V3 + Gender ===")
    print("추움: 33도 미만, 쾌적: 33-35도, 더움: 35도 초과")
    print("사용 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, Gender")
    
    # 1. 데이터 로드 및 전처리
    df = load_and_preprocess_data('extracted_data_sampled_20rows.csv')
    
    # 2. 온도 라벨 생성
    df = create_temperature_labels_v3(df)
    
    # 3. 특성 준비 (Gender 포함)
    X, y, feature_columns = prepare_features_v3_gender(df)
    
    # 4. 모델 학습
    model, X_train, X_val, y_train, y_val = train_catboost_model(X, y, feature_columns)
    
    # 5. 모델 평가
    accuracy, feature_importance_df = evaluate_model(model, X_val, y_val, feature_columns)
    
    # 6. 결과 시각화
    plot_results(model, X_val, y_val, feature_importance_df)
    
    # 7. 상세 분류 분석
    create_detailed_classification_analysis(model, X_val, y_val, feature_columns, feature_importance_df)
    
    # 8. 모델 저장
    model.save_model('temperature_prediction_model_v3_gender.cbm')
    print("\n모델이 'temperature_prediction_model_v3_gender.cbm'으로 저장되었습니다.")
    
    return model, accuracy, feature_importance_df

if __name__ == "__main__":
    model, accuracy, feature_importance = main()
