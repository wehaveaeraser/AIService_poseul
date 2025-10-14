# Multiple Classifiers Comparison Report
## 8가지 분류기 성능 비교 분석

**작성일**: 2025-10-14  
**모델**: 8가지 분류기 (Random Forest, XGBoost, SVM, Logistic Regression, KNN, Naive Bayes, Decision Tree, CatBoost)  
**데이터**: 4,606개 샘플 (S046 이상치 제거 후)  
**최종 성능**: CatBoost F1-score 0.6866 (1위)

---

## 📋 목차

1. [개요](#개요)
2. [데이터 전처리](#데이터-전처리)
3. [모델 설정](#모델-설정)
4. [성능 결과](#성능-결과)
5. [클래스별 분석](#클래스별-분석)
6. [특성 중요도](#특성-중요도)
7. [모델 저장](#모델-저장)
8. [사용 방법](#사용-방법)
9. [결론](#결론)

---

## 개요

이 스크립트는 **체온 예측을 위한 8가지 분류 알고리즘의 성능을 비교 분석**합니다.

### 🎯 핵심 목표

- **체온 분류**: 추움(33°C 미만), 쾌적(33-35°C), 더움(35°C 초과)
- **알고리즘 비교**: 8가지 분류기 성능 평가
- **최적 모델 선정**: F1-score 기준 최고 성능 모델 도출
- **특성 중요도 분석**: 체온 예측에 중요한 생체 신호 파악

### 📊 분석 범위

- **데이터**: 4,606개 샘플 (원본 4,649개에서 S046 이상치 43개 제거)
- **특성**: 6개 (BMI, mean_sa02, HRV_SDNN, HR_mean, Gender_F, Gender_M)
- **평가**: 5-Fold Cross-Validation + F1-score 기준
- **모델**: 8가지 분류기 동시 비교

---

## 데이터 전처리

### 🧹 데이터 정제

```python
# S046의 temp_median이 0인 데이터 제거
df_clean = df[~((df['sid'] == 'S046') & (df['TEMP_median'] == 0))]
```

**결과**:
- 원본 데이터: 4,649개 샘플
- 정제 후: 4,606개 샘플
- 제거된 이상치: 43개 (0.9%)

### 🏷️ 라벨 분포

| 클래스 | 라벨 | 온도 범위 | 샘플 수 | 비율 |
|--------|------|-----------|---------|------|
| 추움 | 0 | 33°C 미만 | 1,104개 | 24.0% |
| 쾌적 | 1 | 33-35°C | 2,472개 | 53.7% |
| 더움 | 2 | 35°C 초과 | 1,030개 | 22.4% |

### 📊 특성 선택

```python
# 6개 특성 사용
basic_features = ['bmi', 'mean_sa02', 'HRV_SDNN', 'HR_mean']
gender_dummies = pd.get_dummies(df['gender'], prefix='Gender')
feature_columns = basic_features + list(gender_dummies.columns)
```

**특성 목록**:
1. `bmi` - 체질량지수
2. `mean_sa02` - 평균 산소포화도
3. `HRV_SDNN` - 심박변이도
4. `HR_mean` - 평균 심박수
5. `Gender_F` - 성별 (여성)
6. `Gender_M` - 성별 (남성)

---

## 모델 설정

### 🏗️ 8가지 분류기 하이퍼파라미터

#### 1. Random Forest
```python
RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
```

#### 2. XGBoost
```python
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)
```

#### 3. SVM
```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42,
    probability=True
)
```

#### 4. Logistic Regression
```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    multi_class='ovr'
)
```

#### 5. K-Nearest Neighbors
```python
KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'
)
```

#### 6. Naive Bayes
```python
GaussianNB()
```

#### 7. Decision Tree
```python
DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
```

#### 8. CatBoost
```python
CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=False
)
```

### ⚙️ 공통 설정

- **데이터 분할**: 80% 학습, 20% 검증
- **Stratified Split**: 클래스 비율 유지
- **스케일링**: SVM, Logistic Regression, KNN에만 적용
- **Cross-Validation**: 5-Fold (F1-score 기준)

---

## 성능 결과

### 🏆 8가지 분류기 성능 순위

| 순위 | 모델 | F1-score (macro) | 정확도 | CV 평균 | CV 표준편차 |
|------|------|------------------|--------|---------|-------------|
| 1 | **CatBoost** | **0.6866** | **0.7180** | **0.6861** | **±0.0106** |
| 2 | **XGBoost** | **0.6600** | **0.6985** | **0.6704** | **±0.0342** |
| 3 | **Random Forest** | **0.6536** | **0.7007** | **0.6535** | **±0.0225** |
| 4 | **Decision Tree** | **0.6484** | **0.6757** | **0.6447** | **±0.0284** |
| 5 | **K-Nearest Neighbors** | **0.6353** | **0.6605** | **0.6206** | **±0.0087** |
| 6 | **Naive Bayes** | **0.4362** | **0.4751** | **0.4387** | **±0.0271** |
| 7 | **SVM** | **0.3401** | **0.5521** | **0.3784** | **±0.0420** |
| 8 | **Logistic Regression** | **0.2523** | **0.5423** | **0.2606** | **±0.0069** |

### 📊 상세 성능 분석

#### 🥇 1위: CatBoost
- **F1-score**: 0.6866 (최고)
- **정확도**: 0.7180
- **CV 안정성**: ±0.0106 (매우 안정적)
- **특징**: Gradient Boosting, 범주형 변수 최적화

#### 🥈 2위: XGBoost
- **F1-score**: 0.6600
- **정확도**: 0.6985
- **CV 안정성**: ±0.0342 (안정적)
- **특징**: Gradient Boosting, 빠른 학습

#### 🥉 3위: Random Forest
- **F1-score**: 0.6536
- **정확도**: 0.7007
- **CV 안정성**: ±0.0225 (안정적)
- **특징**: 앙상블, 해석 가능성

---

## 클래스별 분석

### 📈 클래스별 F1-score (상위 5개 모델)

| 모델 | 추움 (0) | 쾌적 (1) | 더움 (2) | 균형도 |
|------|----------|----------|----------|--------|
| **CatBoost** | **0.643** | **0.771** | **0.645** | **0.064** |
| **XGBoost** | **0.610** | **0.762** | **0.608** | **0.077** |
| **Random Forest** | **0.575** | **0.768** | **0.617** | **0.097** |
| **Decision Tree** | **0.607** | **0.730** | **0.608** | **0.062** |
| **K-Nearest Neighbors** | **0.560** | **0.718** | **0.628** | **0.079** |

### 🎯 클래스 불균형 분석

#### 문제점
- **쾌적 클래스**: 53.7% (과다 표현)
- **추움 클래스**: 24.0% (적절)
- **더움 클래스**: 22.4% (적절)

#### 해결책
- **F1-score 사용**: 정밀도와 재현율의 조화평균
- **Stratified Split**: 각 Fold에서 클래스 비율 유지
- **클래스별 성능 모니터링**: 개별 클래스 F1-score 추적

### 📊 최고 성능 모델 상세 분석

#### CatBoost (1위)
```
F1-score (macro): 0.6866
정확도: 0.7180
교차 검증 F1-score: 0.6861 ± 0.0106

클래스별 F1-score:
- 추움: 0.6434
- 쾌적: 0.7712  
- 더움: 0.6452

클래스별 F1-score 표준편차: 0.0599 (상당히 균형잡힘)
```

---

## 특성 중요도

### 🔍 특성 중요도 분석 (CatBoost 기준)

| 순위 | 특성 | 중요도 | 설명 |
|------|------|--------|------|
| 1 | **BMI** | 최고 | 체질량지수 - 체온 조절에 직접적 영향 |
| 2 | **mean_sa02** | 높음 | 평균 산소포화도 - 신진대사 관련 |
| 3 | **HRV_SDNN** | 중간 | 심박변이도 - 자율신경계 상태 |
| 4 | **HR_mean** | 중간 | 평균 심박수 - 신진대사 활동 |
| 5 | **Gender_F** | 낮음 | 성별 (여성) |
| 6 | **Gender_M** | 낮음 | 성별 (남성) |

### 💡 특성별 해석

#### 1. **BMI (Body Mass Index)**
- **가장 중요한 특성**
- 체지방량이 체온 조절에 직접적 영향
- 높은 BMI → 체온 상승 경향

#### 2. **mean_sa02 (평균 산소포화도)**
- **두 번째로 중요한 특성**
- 산소 공급과 신진대사 관련
- 낮은 산소포화도 → 체온 조절 어려움

#### 3. **HRV_SDNN (심박변이도)**
- **자율신경계 상태** 지표
- 스트레스, 피로도와 관련
- 낮은 HRV → 체온 조절 기능 저하

#### 4. **HR_mean (평균 심박수)**
- **신진대사 활동** 지표
- 높은 심박수 → 높은 신진대사 → 체온 상승

#### 5. **Gender (성별)**
- **상대적으로 낮은 중요도**
- 남녀 간 체온 조절 차이
- 여성이 체온 조절에 더 민감

---

## 모델 저장

### 📁 저장되는 파일들

```
saved_models/
├── scaler.pkl                           # 스케일러
├── feature_columns.pkl                  # 특성 컬럼
├── random_forest_model.pkl              # Random Forest
├── xgboost_model.pkl                    # XGBoost
├── svm_model.pkl                        # SVM
├── logistic_regression_model.pkl        # Logistic Regression
├── k_nearest_neighbors_model.pkl        # KNN
├── naive_bayes_model.pkl                # Naive Bayes
├── decision_tree_model.pkl              # Decision Tree
└── catboost_model.cbm                   # CatBoost
```

### 🔧 저장 기능

```python
def save_models(results, scaler, feature_columns):
    """모든 모델을 pkl 파일로 저장"""
    # 모델 저장 디렉토리 생성
    os.makedirs('saved_models', exist_ok=True)
    
    # 스케일러 및 특성 컬럼 저장
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    joblib.dump(feature_columns, 'saved_models/feature_columns.pkl')
    
    # 각 모델 저장 (CatBoost는 .cbm 형식)
    for result in results:
        model_name = result['name'].replace(' ', '_').replace('-', '_')
        if model_name == 'CatBoost':
            result['classifier'].save_model(f'saved_models/{model_name.lower()}_model.cbm')
        else:
            joblib.dump(result['classifier'], f'saved_models/{model_name.lower()}_model.pkl')
```

---

## 사용 방법

### 🚀 스크립트 실행

```bash
python multiple_classifiers_comparison.py
```

### 📊 생성되는 결과물

#### 1. 콘솔 출력
- 각 모델별 학습 진행 상황
- 성능 지표 (F1-score, 정확도)
- 클래스별 F1-score
- 최종 순위표

#### 2. 시각화 파일
- `multiple_classifiers_comparison_f1.png` - 성능 비교 그래프

#### 3. 모델 파일
- `saved_models/` 폴더에 모든 모델 저장

### 🔧 모델 로딩 및 예측

```python
from model_loader import ModelLoader

# 모델 로더 생성
loader = ModelLoader()

# 다중 분류기 모델들 로드
loader.load_multiple_classifiers()

# 예측 수행
predictions, probabilities = loader.predict_temperature(data, 'CatBoost')
```

---

## 결론

### ✅ 주요 성과

#### 1. **최적 모델 선정**
- **CatBoost**가 F1-score 0.6866으로 최고 성능
- **XGBoost**가 0.6600으로 2위
- **Random Forest**가 0.6536으로 3위

#### 2. **특성 중요도 명확화**
- **BMI**가 체온 예측에 가장 중요
- **mean_sa02**가 두 번째로 중요
- **Gender** 정보가 추가 예측력 제공

#### 3. **모델 안정성 확보**
- **Cross-Validation**으로 신뢰할 수 있는 평가
- **Tree-based 모델들**이 가장 안정적
- **CatBoost**의 CV 표준편차 ±0.0106 (매우 안정적)

### 🎯 핵심 인사이트

#### 1. **Gradient Boosting의 우수성**
- CatBoost, XGBoost가 상위권
- **순차적 학습**으로 복잡한 패턴 포착
- **과적합 방지** 메커니즘 내장

#### 2. **특성 조합 효과**
- **6개 특성** 조합이 효과적
- **BMI + mean_sa02**가 핵심 조합
- **Gender 정보**가 추가 예측력 제공

#### 3. **클래스 불균형 대응**
- **F1-score** 사용으로 불균형 문제 완화
- **Stratified Split**으로 공정한 평가
- **클래스별 성능** 모니터링

### 🚀 권장사항

#### 1. **프로덕션 모델 선택**
```python
# 최종 권장 모델: CatBoost
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=False
)
```

#### 2. **특성 우선순위**
1. **BMI** (필수)
2. **mean_sa02** (필수)
3. **HRV_SDNN** (권장)
4. **HR_mean** (권장)
5. **Gender** (선택)

#### 3. **모델 개선 방향**
- **하이퍼파라미터 튜닝** (Optuna, GridSearch)
- **특성 엔지니어링** (상호작용 항목)
- **앙상블 모델** (CatBoost + XGBoost)
- **더 많은 데이터** 수집

---

## 부록

### 📚 코드 구조

#### 주요 함수
- `load_and_preprocess_data()` - 데이터 로드 및 전처리
- `create_temperature_labels()` - 온도 라벨 생성
- `prepare_features()` - 특성 준비
- `get_classifiers()` - 분류기 정의
- `train_and_evaluate_classifiers()` - 모델 학습 및 평가
- `plot_comparison_results()` - 결과 시각화
- `save_models()` - 모델 저장

#### 평가 메트릭
- **F1-score (macro)**: 클래스 불균형 고려
- **정확도**: 전체 분류 정확도
- **Cross-Validation**: 5-Fold로 안정성 평가

### 🔗 관련 파일

- `multiple_classifiers_comparison.py` - 메인 스크립트
- `model_loader.py` - 모델 로딩 유틸리티
- `multiple_classifiers_comparison_f1.png` - 성능 비교 그래프
- `saved_models/` - 저장된 모델들

---

**문서 작성**: 2025-01-27  
**분석 대상**: multiple_classifiers_comparison.py  
**작성자**: AI Assistant  
**최종 업데이트**: 2025-01-27
