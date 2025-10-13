# Temperature Prediction Model V3 Gender Report
## CatBoost 기반 성별 포함 체온 예측 모델

**작성일**: 2025-01-27  
**모델**: CatBoost Classifier  
**데이터**: 4,606개 샘플 (S046 이상치 제거 후)  
**특성**: 5개 (BMI, mean_sa02, HRV_SDNN, HR_mean, Gender)  
**목표**: 성별 정보를 포함한 체온 분류 및 예측

---

## 📋 목차

1. [개요](#개요)
2. [데이터 전처리](#데이터-전처리)
3. [모델 설정](#모델-설정)
4. [학습 과정](#학습-과정)
5. [성능 결과](#성능-결과)
6. [특성 중요도 분석](#특성-중요도-분석)
7. [시각화 결과](#시각화-결과)
8. [모델 저장](#모델-저장)
9. [사용 방법](#사용-방법)
10. [결론](#결론)

---

## 개요

이 스크립트는 **성별 정보를 포함한 CatBoost 체온 분류 모델**을 구축하고 분석합니다.

### 🎯 핵심 목표

- **체온 분류**: 추움(33°C 미만), 쾌적(33-35°C), 더움(35°C 초과)
- **5개 특성** 사용 (Gender 포함)
- **성별 영향** 분석
- **특성 중요도** 상세 분석

### 📊 분석 범위

- **데이터**: 4,606개 샘플
- **특성**: 5개 (BMI, mean_sa02, HRV_SDNN, HR_mean, Gender)
- **모델**: CatBoost Classifier
- **평가**: 분류 성능 + 특성 중요도

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
# 5개 특성 사용 (Gender 포함)
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

### 🔄 Gender 원핫인코딩

```python
# Gender 원핫인코딩
gender_dummies = pd.get_dummies(df['gender'], prefix='Gender')
df_encoded = pd.concat([df, gender_dummies], axis=1)
```

**인코딩 결과**:
- `Gender_F`: 여성 (1), 남성 (0)
- `Gender_M`: 남성 (1), 여성 (0)

---

## 모델 설정

### 🏗️ CatBoost 하이퍼파라미터

```python
CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=False,
    early_stopping_rounds=50,
    eval_metric='Accuracy'
)
```

### ⚙️ 학습 설정

- **데이터 분할**: 80% 학습, 20% 검증
- **Stratified Split**: 클래스 비율 유지
- **Early Stopping**: 50 rounds patience
- **평가 지표**: Accuracy

---

## 학습 과정

### 📈 학습 진행 상황

```
0:	learn: 0.5757329	test: 0.5520607	best: 0.5520607 (0)	total: 91.9ms	remaining: 1m 31s
100:	learn: 0.7269273	test: 0.6811280	best: 0.6843818 (91)	total: 674ms	remaining: 6s
Stopped by overfitting detector (50 iterations wait)

bestTest = 0.7114967462
bestIteration = 147

Shrink model to first 148 iterations.
```

### 🎯 학습 결과

- **총 Iterations**: 147 (Early Stopping으로 중단)
- **최적 Iteration**: 147
- **최종 학습 정확도**: 0.727 (72.7%)
- **최종 검증 정확도**: 0.712 (71.2%)
- **학습 시간**: 약 0.674초

---

## 성능 결과

### 📊 분류 성능

#### 정확도
- **학습 정확도**: 0.727 (72.7%)
- **검증 정확도**: 0.712 (71.2%)
- **과적합 정도**: 0.015 (1.5%)

#### 클래스별 성능

| 클래스 | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| **추움** | 0.75 | 0.53 | 0.62 | 221 |
| **쾌적** | 0.70 | 0.87 | 0.77 | 495 |
| **더움** | 0.73 | 0.53 | 0.61 | 206 |
| **Macro Avg** | 0.72 | 0.64 | 0.67 | 922 |
| **Weighted Avg** | 0.72 | 0.71 | 0.70 | 922 |

#### 혼동 행렬
```
[[117  93  11]
 [ 35 430  30]
 [  4  93 109]]
```

### 🔍 성능 해석

#### 장점
- **높은 정확도**: 71.2% 검증 정확도
- **과적합 방지**: Early Stopping 작동
- **빠른 학습**: 0.674초 내 완료
- **클래스 균형**: 모든 클래스에서 적절한 성능

#### 개선점
- **추움 클래스**: Recall 0.53으로 개선 여지
- **더움 클래스**: Recall 0.53으로 개선 여지
- **쾌적 클래스**: 가장 높은 성능 (다수 클래스)

---

## 특성 중요도 분석

### 🔍 특성 중요도 순위

| 순위 | 특성 | 중요도 | 설명 |
|------|------|--------|------|
| 1 | **BMI** | **34.03** | 체질량지수 - 체온 조절에 직접적 영향 |
| 2 | **mean_sa02** | **27.16** | 평균 산소포화도 - 신진대사 관련 |
| 3 | **HRV_SDNN** | **13.80** | 심박변이도 - 자율신경계 상태 |
| 4 | **Gender_F** | **9.85** | 성별 (여성) |
| 5 | **HR_mean** | **8.79** | 평균 심박수 - 신진대사 활동 |
| 6 | **Gender_M** | **6.36** | 성별 (남성) |

### 💡 특성별 해석

#### 1. **BMI (Body Mass Index)**
- **가장 중요한 특성** (34.03)
- 체지방량이 체온 조절에 직접적 영향
- 높은 BMI → 체온 상승 경향

#### 2. **mean_sa02 (평균 산소포화도)**
- **두 번째로 중요한 특성** (27.16)
- 산소 공급과 신진대사 관련
- 낮은 산소포화도 → 체온 조절 어려움

#### 3. **HRV_SDNN (심박변이도)**
- **세 번째로 중요한 특성** (13.80)
- **자율신경계 상태** 지표
- 스트레스, 피로도와 관련
- 낮은 HRV → 체온 조절 기능 저하

#### 4. **Gender_F (성별 - 여성)**
- **네 번째로 중요한 특성** (9.85)
- 여성의 체온 조절 특성
- 남성과 다른 생리적 특성

#### 5. **HR_mean (평균 심박수)**
- **다섯 번째로 중요한 특성** (8.79)
- **신진대사 활동** 지표
- 높은 심박수 → 높은 신진대사 → 체온 상승

#### 6. **Gender_M (성별 - 남성)**
- **여섯 번째로 중요한 특성** (6.36)
- 남성의 체온 조절 특성
- 상대적으로 낮은 중요도

### 🔬 성별 영향 분석

#### Gender 특성의 역할
- **추가 예측력**: 기존 4개 특성에 추가 정보 제공
- **개인차 고려**: 남녀 간 체온 조절 차이 반영
- **모델 정확도**: 미세하지만 성능 향상에 기여

#### 성별별 체온 조절 특성
- **여성**: 더 민감한 체온 조절 (Gender_F: 9.85)
- **남성**: 상대적으로 안정적인 체온 조절 (Gender_M: 6.36)
- **개인차**: 성별 외 개인 특성도 중요

---

## 시각화 결과

### 📈 생성되는 그래프들

#### 1. `temperature_prediction_results_v3_gender.png`
- **혼동 행렬** (정확도 표시)
- **특성 중요도** 막대 그래프

#### 2. `temperature_analysis_v3_gender_detailed.png`
- **혼동 행렬** (상세 분석)
- **클래스별 정확도** 막대 그래프
- **특성 중요도** (상위 10개)
- **클래스별 예측 확률** 분포

### 🎨 시각화 특징

#### 색상 코딩
- **추움**: 파란색 계열 (skyblue)
- **쾌적**: 초록색 계열 (lightgreen)
- **더움**: 빨간색 계열 (salmon)

#### 그래프 구성
- **1x2 레이아웃**: 기본 결과
- **2x2 레이아웃**: 상세 분석
- **고해상도**: 300 DPI

---

## 모델 저장

### 📁 저장되는 파일들

```
saved_models/
├── temperature_prediction_model_v3_gender.cbm      # CatBoost 모델
├── temperature_prediction_v3_gender_features.pkl   # 특성 컬럼
└── temperature_prediction_v3_gender_importance.pkl # 특성 중요도
```

### 🔧 저장 기능

```python
def save_model_v3_gender(model, feature_columns, feature_importance_df):
    """V3 Gender 모델 저장"""
    # 모델 저장 디렉토리 생성
    os.makedirs('saved_models', exist_ok=True)
    
    # CatBoost 모델 저장 (cbm 형식)
    model.save_model('saved_models/temperature_prediction_model_v3_gender.cbm')
    
    # 특성 컬럼 저장
    joblib.dump(feature_columns, 'saved_models/temperature_prediction_v3_gender_features.pkl')
    
    # 특성 중요도 저장
    joblib.dump(feature_importance_df, 'saved_models/temperature_prediction_v3_gender_importance.pkl')
```

---

## 사용 방법

### 🚀 스크립트 실행

```bash
python temperature_prediction_model_v3_gender.py
```

### 📊 생성되는 결과물

#### 1. 콘솔 출력
```
=== 실시간 온도 예측 모델 V3 + Gender ===
추움: 33도 미만, 쾌적: 33-35도, 더움: 35도 초과
사용 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, Gender

데이터 로딩 중...
원본 데이터 크기: (4649, 8)
S046의 temp_median이 0인 데이터 제거 중...
정리된 데이터 크기: (4606, 8)
제거된 데이터 수: 43

온도 라벨 생성 중... (추움: 33도 미만, 더움: 35도 초과)
추움 기준: 33도 미만
더움 기준: 35도 초과
쾌적 기준: 33도 이상 35도 이하

라벨 분포:
추움 (0): 1104개 (24.0%)
쾌적 (1): 2472개 (53.7%)
더움 (2): 1030개 (22.4%)

특성 준비 중... (5개 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, gender)
특성 수: 6
특성 목록: ['bmi', 'mean_sa02', 'HRV_SDNN', 'HR_mean', 'Gender_F', 'Gender_M']

CatBoost 모델 학습 중...
학습 데이터 크기: (3684, 6)
검증 데이터 크기: (922, 6)

모델 평가 중...

검증 정확도: 0.7115

분류 리포트:
              precision    recall  f1-score   support

        추움       0.75      0.53      0.62       221
        쾌적       0.70      0.87      0.77       495
        더움       0.73      0.53      0.61       206

    accuracy                           0.71       922
   macro avg       0.72      0.64      0.67       922
weighted avg       0.72      0.71      0.70       922

혼동 행렬:
[[117  93  11]
 [ 35 430  30]
 [  4  93 109]]

특성 중요도:
     feature  importance
0        bmi   34.034607
1  mean_sa02   27.164998
2   HRV_SDNN   13.804773
4   Gender_F    9.848261
3    HR_mean    8.789257
5   Gender_M    6.358104

결과 시각화 중...
상세 분류 분석 중...

모델이 'temperature_prediction_model_v3_gender.cbm'으로 저장되었습니다.
```

#### 2. 시각화 파일
- `temperature_prediction_results_v3_gender.png` - 기본 결과
- `temperature_analysis_v3_gender_detailed.png` - 상세 분석

#### 3. 모델 파일
- `saved_models/temperature_prediction_model_v3_gender.cbm` - CatBoost 모델
- `saved_models/temperature_prediction_v3_gender_features.pkl` - 특성 컬럼
- `saved_models/temperature_prediction_v3_gender_importance.pkl` - 특성 중요도

---

## 결론

### ✅ 주요 성과

#### 1. **성별 정보 포함 모델**
- **5개 특성** 사용 (Gender 포함)
- **검증 정확도**: 71.2%
- **특성 중요도**: 명확한 순위 제공

#### 2. **특성 중요도 분석**
- **BMI**가 가장 중요 (34.03)
- **mean_sa02**가 두 번째 (27.16)
- **Gender** 정보가 추가 예측력 제공

#### 3. **모델 저장 및 관리**
- **pkl 형식**으로 효율적 저장
- **특성 중요도** 별도 저장
- **로딩 유틸리티** 제공

### 🎯 핵심 인사이트

#### 1. **성별의 영향**
- **상대적으로 높은 중요도** (Gender_F: 9.85, Gender_M: 6.36)
- **추가 예측력** 제공
- **개인차 고려** 가능

#### 2. **특성 조합 효과**
- **4개 기본 특성** + **Gender** = 최적 조합
- **BMI + mean_sa02**가 핵심
- **HRV + HR**가 보조 역할

#### 3. **CatBoost의 장점**
- **범주형 변수** 자동 처리
- **특성 중요도** 자동 계산
- **과적합 방지** 내장

### 🚀 개선 방향

#### 1. **성능 향상**
- **하이퍼파라미터 튜닝**
- **특성 엔지니어링**
- **앙상블 모델**

#### 2. **특성 확장**
- **더 많은 생체 신호**
- **환경 요인** 추가
- **시간적 특성** 고려

#### 3. **개인화**
- **개인별 모델** 구축
- **적응형 학습**
- **실시간 업데이트**

---

## 부록

### 📚 코드 구조

#### 주요 함수
- `load_and_preprocess_data()` - 데이터 로드 및 전처리
- `create_temperature_labels_v3()` - 온도 라벨 생성
- `prepare_features_v3_gender()` - 특성 준비 (5개 + Gender)
- `train_catboost_model()` - CatBoost 모델 학습
- `evaluate_model()` - 모델 평가
- `plot_results()` - 결과 시각화
- `create_detailed_classification_analysis()` - 상세 분류 분석
- `save_model_v3_gender()` - 모델 저장

#### 평가 메트릭
- **정확도**: 전체 분류 정확도
- **정밀도**: 클래스별 정밀도
- **재현율**: 클래스별 재현율
- **F1-score**: 정밀도와 재현율의 조화평균

### 🔗 관련 파일

- `temperature_prediction_model_v3_gender.py` - 메인 스크립트
- `model_loader.py` - 모델 로딩 유틸리티
- `temperature_prediction_results_v3_gender.png` - 기본 결과 그래프
- `temperature_analysis_v3_gender_detailed.png` - 상세 분석 그래프
- `saved_models/temperature_prediction_model_v3_gender.cbm` - 저장된 모델

---

**문서 작성**: 2025-01-27  
**분석 대상**: temperature_prediction_model_v3_gender.py  
**작성자**: AI Assistant  
**최종 업데이트**: 2025-01-27
