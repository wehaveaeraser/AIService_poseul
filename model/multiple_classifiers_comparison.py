import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

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

def create_temperature_labels(df):
    """온도 라벨 생성"""
    print("온도 라벨 생성 중... (추움: 33도 미만, 더움: 35도 초과)")
    
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

def prepare_features(df):
    """특성 준비 (6개 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, Gender)"""
    print("특성 준비 중... (6개 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, Gender)")
    
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

def get_classifiers():
    """다양한 분류기 정의"""
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        'CatBoost': CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
    }
    return classifiers

def train_and_evaluate_classifiers(X, y, feature_columns):
    """모든 분류기 학습 및 평가"""
    print("다양한 분류기 학습 및 평가 중...")
    
    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 스케일링 (SVM, Logistic Regression, KNN에 필요)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    classifiers = get_classifiers()
    results = []
    
    for name, classifier in classifiers.items():
        print(f"\n--- {name} 학습 중 ---")
        
        try:
            # 스케일링이 필요한 모델들
            if name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
                X_train_use = X_train_scaled
                X_val_use = X_val_scaled
            else:
                X_train_use = X_train
                X_val_use = X_val
            
            # 교차 검증 (F1-score 기준)
            cv_scores = cross_val_score(classifier, X_train_use, y_train, cv=5, scoring='f1_macro')
            
            # 모델 학습
            classifier.fit(X_train_use, y_train)
            
            # 예측
            y_pred = classifier.predict(X_val_use)
            
            # 정확도 및 F1-score 계산
            accuracy = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average='macro')
            f1_weighted = f1_score(y_val, y_pred, average='weighted')
            
            # 클래스별 F1-score
            f1_per_class = f1_score(y_val, y_pred, average=None)
            
            # 클래스별 정확도
            class_accuracies = []
            for i in range(3):
                class_mask = y_val == i
                if np.sum(class_mask) > 0:
                    class_acc = accuracy_score(y_val[class_mask], y_pred[class_mask])
                    class_accuracies.append(class_acc)
                else:
                    class_accuracies.append(0)
            
            # 특성 중요도 (가능한 경우)
            feature_importance = None
            if hasattr(classifier, 'feature_importances_'):
                feature_importance = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                # Logistic Regression의 경우 절댓값 사용
                feature_importance = np.abs(classifier.coef_).mean(axis=0)
            
            result = {
                'name': name,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'val_accuracy': accuracy,
                'val_f1_macro': f1_macro,
                'val_f1_weighted': f1_weighted,
                'f1_per_class': f1_per_class,
                'class_accuracies': class_accuracies,
                'feature_importance': feature_importance,
                'y_pred': y_pred,
                'classifier': classifier
            }
            
            results.append(result)
            
            print(f"교차 검증 F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"검증 정확도: {accuracy:.4f}")
            print(f"검증 F1-score (macro): {f1_macro:.4f}")
            print(f"클래스별 F1-score - 추움: {f1_per_class[0]:.3f}, 쾌적: {f1_per_class[1]:.3f}, 더움: {f1_per_class[2]:.3f}")
            
        except Exception as e:
            print(f"{name} 학습 중 오류 발생: {str(e)}")
            continue
    
    return results, X_val, y_val, feature_columns

def plot_comparison_results(results, X_val, y_val, feature_columns):
    """결과 비교 시각화"""
    print("결과 비교 시각화 중...")
    
    # 결과 정리
    model_names = [r['name'] for r in results]
    cv_scores = [r['cv_mean'] for r in results]
    val_scores = [r['val_accuracy'] for r in results]
    f1_scores = [r['val_f1_macro'] for r in results]
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. F1-score vs 정확도 비교
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    axes[0,0].bar(x_pos - width/2, f1_scores, width, label='F1-score (macro)', alpha=0.8, color='skyblue')
    axes[0,0].bar(x_pos + width/2, val_scores, width, label='정확도', alpha=0.8, color='lightcoral')
    axes[0,0].set_xlabel('모델', fontsize=12)
    axes[0,0].set_ylabel('점수', fontsize=12)
    axes[0,0].set_title('F1-score vs 정확도 비교', fontsize=14, fontweight='bold')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    axes[0,0].legend(fontsize=11)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1.0)
    
    # 막대 위에 값 표시 (겹침 방지)
    for i, (f1, val) in enumerate(zip(f1_scores, val_scores)):
        # F1-score 값
        axes[0,0].text(i - width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', 
                      fontsize=8, fontweight='bold')
        # 정확도 값
        axes[0,0].text(i + width/2, val + 0.01, f'{val:.3f}', ha='center', va='bottom', 
                      fontsize=8, fontweight='bold')
    
    # 2. 클래스별 F1-score (상위 5개 모델)
    top_5_indices = np.argsort(f1_scores)[-5:]
    top_5_names = [model_names[i] for i in top_5_indices]
    top_5_f1 = [results[i]['f1_per_class'] for i in top_5_indices]
    
    x_pos = np.arange(len(top_5_names))
    width = 0.25
    
    bars1 = axes[0,1].bar(x_pos - width, [f1[0] for f1 in top_5_f1], width, label='추움', alpha=0.8, color='skyblue')
    bars2 = axes[0,1].bar(x_pos, [f1[1] for f1 in top_5_f1], width, label='쾌적', alpha=0.8, color='lightgreen')
    bars3 = axes[0,1].bar(x_pos + width, [f1[2] for f1 in top_5_f1], width, label='더움', alpha=0.8, color='salmon')
    
    axes[0,1].set_xlabel('모델', fontsize=12)
    axes[0,1].set_ylabel('F1-score', fontsize=12)
    axes[0,1].set_title('상위 5개 모델의 클래스별 F1-score', fontsize=14, fontweight='bold')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(top_5_names, rotation=45, ha='right', fontsize=9)
    axes[0,1].legend(fontsize=11)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1.1)  # 여백 확보
    
    # 막대 위에 값 표시 (겹침 방지)
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
        height1, height2, height3 = bar1.get_height(), bar2.get_height(), bar3.get_height()
        if height1 > 0.1:  # 값이 너무 작으면 표시하지 않음
            axes[0,1].text(bar1.get_x() + bar1.get_width()/2, height1 + 0.02, f'{height1:.3f}', 
                          ha='center', va='bottom', fontsize=7, fontweight='bold')
        if height2 > 0.1:
            axes[0,1].text(bar2.get_x() + bar2.get_width()/2, height2 + 0.02, f'{height2:.3f}', 
                          ha='center', va='bottom', fontsize=7, fontweight='bold')
        if height3 > 0.1:
            axes[0,1].text(bar3.get_x() + bar3.get_width()/2, height3 + 0.02, f'{height3:.3f}', 
                          ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # 3. 특성 중요도 비교 (상위 3개 모델)
    top_3_indices = np.argsort(f1_scores)[-3:]
    top_3_names = [model_names[i] for i in top_3_indices]
    
    x_pos = np.arange(len(feature_columns))
    width = 0.25
    
    for i, idx in enumerate(top_3_indices):
        if results[idx]['feature_importance'] is not None:
            importance = results[idx]['feature_importance']
            axes[1,0].bar(x_pos + i*width, importance, width, alpha=0.8, label=top_3_names[i])
    
    axes[1,0].set_xlabel('특성', fontsize=12)
    axes[1,0].set_ylabel('중요도', fontsize=12)
    axes[1,0].set_title('상위 3개 모델의 특성 중요도', fontsize=14, fontweight='bold')
    axes[1,0].set_xticks(x_pos + width)
    axes[1,0].set_xticklabels(feature_columns, rotation=45, ha='right', fontsize=8)
    axes[1,0].legend(fontsize=11)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. F1-score 순위
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    
    bars = axes[1,1].barh(range(len(sorted_names)), sorted_f1, alpha=0.8, color='lightblue')
    axes[1,1].set_yticks(range(len(sorted_names)))
    axes[1,1].set_yticklabels(sorted_names, fontsize=9)
    axes[1,1].set_xlabel('F1-score (macro)', fontsize=12)
    axes[1,1].set_title('모델 F1-score 순위', fontsize=14, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlim(0, 1.0)
    
    # 막대 옆에 값 표시 (겹침 방지)
    for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
        axes[1,1].text(score + 0.008, bar.get_y() + bar.get_height()/2, 
                      f'{score:.3f}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('multiple_classifiers_comparison_f1.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(results):
    """상세 결과 출력"""
    print("\n" + "="*80)
    print("모델 성능 상세 비교")
    print("="*80)
    
    # F1-score 순으로 정렬
    sorted_results = sorted(results, key=lambda x: x['val_f1_macro'], reverse=True)
    
    print(f"{'순위':<4} {'모델명':<20} {'F1(macro)':<10} {'정확도':<10} {'추움F1':<8} {'쾌적F1':<8} {'더움F1':<8}")
    print("-"*80)
    
    for i, result in enumerate(sorted_results, 1):
        f1_macro = f"{result['val_f1_macro']:.4f}"
        accuracy = f"{result['val_accuracy']:.4f}"
        f1_per_class = [f"{f1:.3f}" for f1 in result['f1_per_class']]
        
        print(f"{i:<4} {result['name']:<20} {f1_macro:<10} {accuracy:<10} {f1_per_class[0]:<8} {f1_per_class[1]:<8} {f1_per_class[2]:<8}")
    
    print("\n" + "="*80)
    print("최고 성능 모델 상세 분석")
    print("="*80)
    
    best_model = sorted_results[0]
    print(f"모델: {best_model['name']}")
    print(f"F1-score (macro): {best_model['val_f1_macro']:.4f}")
    print(f"정확도: {best_model['val_accuracy']:.4f}")
    print(f"교차 검증 F1-score: {best_model['cv_mean']:.4f} ± {best_model['cv_std']:.4f}")
    print(f"클래스별 F1-score:")
    print(f"  - 추움: {best_model['f1_per_class'][0]:.4f}")
    print(f"  - 쾌적: {best_model['f1_per_class'][1]:.4f}")
    print(f"  - 더움: {best_model['f1_per_class'][2]:.4f}")
    
    # 클래스 불균형 개선 정도 분석
    f1_std = np.std(best_model['f1_per_class'])
    print(f"\n클래스별 F1-score 표준편차: {f1_std:.4f} (낮을수록 균형잡힘)")
    
    if f1_std < 0.1:
        print("✅ 클래스 간 성능이 매우 균형잡혀 있습니다!")
    elif f1_std < 0.2:
        print("✅ 클래스 간 성능이 상당히 균형잡혀 있습니다.")
    else:
        print("⚠️ 클래스 간 성능 차이가 여전히 있습니다.")

def main():
    """메인 함수"""
    print("=== F1-score 기반 분류 모델 성능 비교 ===")
    print("추움: 33도 미만, 쾌적: 33-35도, 더움: 35도 초과")
    print("사용 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, Gender")
    
    # 1. 데이터 로드 및 전처리
    df = load_and_preprocess_data('extracted_data_sampled_20rows.csv')
    
    # 2. 온도 라벨 생성
    df = create_temperature_labels(df)
    
    # 3. 특성 준비
    X, y, feature_columns = prepare_features(df)
    
    # 4. 모든 분류기 학습 및 평가
    results, X_val, y_val, feature_columns = train_and_evaluate_classifiers(X, y, feature_columns)
    
    # 5. 결과 시각화
    plot_comparison_results(results, X_val, y_val, feature_columns)
    
    # 6. 상세 결과 출력
    print_detailed_results(results)
    
    return results

if __name__ == "__main__":
    results = main()
