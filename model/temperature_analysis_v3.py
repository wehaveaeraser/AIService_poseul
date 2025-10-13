import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
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
    """TEMP_median 기준으로 온도 라벨 생성"""
    print("온도 라벨 생성 중... (추움: 33도 미만, 더움: 35도 초과)")
    
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

def prepare_features_v3(df):
    """특성 준비 (4개 특성만 사용)"""
    print("특성 준비 중... (4개 특성만 사용)")
    
    # 사용할 특성만 선택
    feature_columns = ['HR_mean', 'HRV_SDNN', 'bmi', 'mean_sa02']
    
    X = df[feature_columns]
    y = df['temperature_label']
    
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

def create_detailed_analysis_plots(model, X_val, y_val, df_val):
    """상세한 분석 그래프 생성"""
    print("상세 분석 그래프 생성 중...")
    
    # 예측
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    # 온도 예측을 위한 연속값 변환 (라벨을 온도로 역변환)
    def label_to_temp(label):
        if label == 0:  # 추움
            return 32.0  # 추움의 대표값
        elif label == 1:  # 쾌적
            return 34.0  # 쾌적의 대표값
        else:  # 더움
            return 36.0  # 더움의 대표값
    
    # 실제 온도와 예측 온도
    actual_temp = df_val['TEMP_median'].values
    predicted_temp = np.array([label_to_temp(label) for label in y_pred])
    
    # R² 점수 계산
    r2 = r2_score(actual_temp, predicted_temp)
    rmse = np.sqrt(mean_squared_error(actual_temp, predicted_temp))
    
    # 잔차 계산
    residuals = actual_temp - predicted_temp
    
    # 그래프 생성
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 실제 vs 예측 온도 산점도
    axes[0].scatter(actual_temp, predicted_temp, alpha=0.6, color='blue', s=30)
    axes[0].plot([actual_temp.min(), actual_temp.max()], 
                 [actual_temp.min(), actual_temp.max()], 'r--', lw=2)
    axes[0].set_xlabel('실제 온도 (°C)')
    axes[0].set_ylabel('예측 온도 (°C)')
    axes[0].set_title(f'실제 vs 예측 온도\nR² = {r2:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 잔차 분석
    axes[1].scatter(predicted_temp, residuals, alpha=0.6, color='green', s=30)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('예측 온도 (°C)')
    axes[1].set_ylabel('잔차 (°C)')
    axes[1].set_title(f'잔차 분석\nRMSE = {rmse:.4f}°C')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 온도 분포 비교
    axes[2].hist(actual_temp, bins=20, alpha=0.7, label='실제', color='blue', density=True)
    axes[2].hist(predicted_temp, bins=20, alpha=0.7, label='예측', color='red', density=True)
    axes[2].set_xlabel('온도 (°C)')
    axes[2].set_ylabel('빈도')
    axes[2].set_title('온도 분포 비교')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_analysis_v3_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return r2, rmse

def create_classification_analysis_plots(model, X_val, y_val):
    """분류 분석 그래프 생성"""
    print("분류 분석 그래프 생성 중...")
    
    # 예측
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)
    
    # 혼동 행렬
    cm = confusion_matrix(y_val, y_pred)
    
    # 정확도 계산
    accuracy = accuracy_score(y_val, y_pred)
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 혼동 행렬
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['추움', '쾌적', '더움'],
                yticklabels=['추움', '쾌적', '더움'],
                ax=axes[0,0])
    axes[0,0].set_title(f'혼동 행렬\n정확도: {accuracy:.4f}')
    axes[0,0].set_ylabel('실제')
    axes[0,0].set_xlabel('예측')
    
    # 2. 클래스별 예측 확률 분포
    for i, class_name in enumerate(['추움', '쾌적', '더움']):
        class_proba = y_pred_proba[y_val == i, :]
        axes[0,1].hist(class_proba[:, i], bins=20, alpha=0.6, 
                      label=f'{class_name} (실제)', density=True)
    axes[0,1].set_xlabel('예측 확률')
    axes[0,1].set_ylabel('빈도')
    axes[0,1].set_title('클래스별 예측 확률 분포')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 예측 확률 히트맵 (샘플별)
    sample_indices = np.random.choice(len(y_val), min(100, len(y_val)), replace=False)
    proba_subset = y_pred_proba[sample_indices]
    
    im = axes[1,0].imshow(proba_subset.T, cmap='YlOrRd', aspect='auto')
    axes[1,0].set_xlabel('샘플 인덱스')
    axes[1,0].set_ylabel('클래스')
    axes[1,0].set_title('예측 확률 히트맵 (100개 샘플)')
    axes[1,0].set_yticks([0, 1, 2])
    axes[1,0].set_yticklabels(['추움', '쾌적', '더움'])
    plt.colorbar(im, ax=axes[1,0])
    
    # 4. 클래스별 정확도
    class_names = ['추움', '쾌적', '더움']
    class_accuracies = []
    for i in range(3):
        class_mask = y_val == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_val[class_mask], y_pred[class_mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    bars = axes[1,1].bar(class_names, class_accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1,1].set_ylabel('정확도')
    axes[1,1].set_title('클래스별 정확도')
    axes[1,1].set_ylim(0, 1)
    
    # 막대 위에 정확도 값 표시
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{acc:.3f}', ha='center', va='bottom')
    
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_analysis_v3_classification.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("=== V3 모델 상세 분석 ===")
    print("추움: 33도 미만, 쾌적: 33-35도, 더움: 35도 초과")
    print("사용 특성: HR_mean, HRV_SDNN, BMI, mean_sa02")
    
    # 1. 데이터 로드 및 전처리
    df = load_and_preprocess_data('extracted_data_sampled_20rows.csv')
    
    # 2. 온도 라벨 생성
    df = create_temperature_labels_v3(df)
    
    # 3. 특성 준비
    X, y, feature_columns = prepare_features_v3(df)
    
    # 4. 모델 학습
    model, X_train, X_val, y_train, y_val = train_catboost_model(X, y, feature_columns)
    
    # 5. 검증 데이터에 해당하는 원본 데이터 추출
    df_val = df.iloc[X_val.index]
    
    # 6. 상세 분석 그래프 생성
    r2, rmse = create_detailed_analysis_plots(model, X_val, y_val, df_val)
    
    # 7. 분류 분석 그래프 생성
    create_classification_analysis_plots(model, X_val, y_val)
    
    # 8. 결과 요약
    print(f"\n=== 분석 결과 요약 ===")
    print(f"R² 점수: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}°C")
    
    return model, r2, rmse

if __name__ == "__main__":
    model, r2, rmse = main()
