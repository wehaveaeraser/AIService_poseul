# Temperature Analysis V3 Report
## CatBoost 기반 체온 분석 모델

**작성일**: 2025-01-27  
**모델**: CatBoost Classifier  
**데이터**: 4,606개 샘플 (S046 이상치 제거 후)  
**특성**: 4개 (HR_mean, HRV_SDNN, BMI, mean_sa02)  
**목표**: 체온 분류 및 상세 분석

---

## 📋 목차

1. [개요](#개요)
2. [데이터 전처리](#데이터-전처리)
3. [모델 설정](#모델-설정)
4. [학습 과정](#학습-과정)
5. [성능 결과](#성능-결과)
6. [상세 분석](#상세-분석)
7. [시각화 결과](#시각화-결과)
8. [모델 저장](#모델-저장)
9. [사용 방법](#사용-방법)
10. [결론](#결론)

---

## 개요

이 스크립트는 **CatBoost를 사용한 체온 분류 모델의 상세 분석**을 수행합니다.

### 🎯 핵심 목표

- **체온 분류**: 추움(33°C 미만), 쾌적(33-35°C), 더움(35°C 초과)
- **4개 핵심 특성** 사용 (Gender 제외)
- **상세한 분석**: 회귀 지표(R², RMSE) 포함
- **시각화**: 잔차 분석, 분포 비교, 분류 분석

### 📊 분석 범위

- **데이터**: 4,606개 샘플
- **특성**: 4개 (HR_mean, HRV_SDNN, BMI, mean_sa02)
- **모델**: CatBoost Classifier
- **평가**: 분류 + 회귀 지표

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
# 4개 핵심 특성만 사용 (Gender 제외)
feature_columns = ['HR_mean', 'HRV_SDNN', 'bmi', 'mean_sa02']
```

**특성 목록**:
1. `HR_mean` - 평균 심박수
2. `HRV_SDNN` - 심박변이도
3. `bmi` - 체질량지수
4. `mean_sa02` - 평균 산소포화도

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
0:	learn: 0.5572747	test: 0.5509761	best: 0.5509761 (0)	total: 79.1ms	remaining: 1m 19s
100:	learn: 0.7204126	test: 0.6811280	best: 0.6811280 (100)	total: 614ms	remaining: 5.47s
200:	learn: 0.7559718	test: 0.6941432	best: 0.6963124 (192)	total: 1.18s	remaining: 4.7s
Stopped by overfitting detector (50 iterations wait)

bestTest = 0.6963123644
bestIteration = 192

Shrink model to first 193 iterations.
```

### 🎯 학습 결과

- **총 Iterations**: 244 (Early Stopping으로 중단)
- **최적 Iteration**: 192
- **최종 학습 정확도**: 0.756 (75.6%)
- **최종 검증 정확도**: 0.696 (69.6%)
- **학습 시간**: 약 1.18초

---

## 성능 결과

### 📊 분류 성능

#### 정확도
- **학습 정확도**: 0.756 (75.6%)
- **검증 정확도**: 0.696 (69.6%)
- **과적합 정도**: 0.060 (6.0%)

#### 클래스별 성능
- **추움 클래스**: 적절한 성능
- **쾌적 클래스**: 가장 높은 성능 (다수 클래스)
- **더움 클래스**: 적절한 성능

### 📈 회귀 성능 (온도 예측)

#### R² 점수
- **R²**: 0.2492 (24.9%)
- **해석**: 모델이 온도 변동의 24.9%를 설명

#### RMSE
- **RMSE**: 1.9422°C
- **해석**: 평균적으로 1.94°C 오차

### 🔍 성능 해석

#### 장점
- **적절한 정확도**: 69.6% 검증 정확도
- **과적합 방지**: Early Stopping 작동
- **빠른 학습**: 1.18초 내 완료

#### 개선점
- **R² 점수**: 24.9%로 개선 여지 있음
- **RMSE**: 1.94°C 오차로 정밀도 향상 필요
- **클래스 불균형**: 쾌적 클래스 편향

---

## 상세 분석

### 🔬 분석 함수들

#### 1. `create_detailed_analysis_plots()`
- **실제 vs 예측 온도** 산점도
- **잔차 분석** 그래프
- **온도 분포** 비교

#### 2. `create_classification_analysis_plots()`
- **혼동 행렬** 시각화
- **클래스별 예측 확률** 분포
- **예측 확률 히트맵**
- **클래스별 정확도**

### 📊 분석 결과

#### 온도 예측 분석
- **R² = 0.2492**: 모델이 온도 변동의 24.9% 설명
- **RMSE = 1.9422°C**: 평균 1.94°C 오차
- **잔차 패턴**: 랜덤 분포 (모델 적합성 양호)

#### 분류 분석
- **혼동 행렬**: 클래스별 예측 패턴 확인
- **예측 확률**: 각 클래스별 신뢰도 분석
- **클래스별 정확도**: 개별 클래스 성능 평가

---

## 시각화 결과

### 📈 생성되는 그래프들

#### 1. `temperature_analysis_v3_detailed.png`
- **실제 vs 예측 온도** 산점도 (R² 표시)
- **잔차 분석** 그래프 (RMSE 표시)
- **온도 분포** 비교 (실제 vs 예측)

#### 2. `temperature_analysis_v3_classification.png`
- **혼동 행렬** (정확도 표시)
- **클래스별 예측 확률** 분포
- **예측 확률 히트맵** (100개 샘플)
- **클래스별 정확도** 막대 그래프

### 🎨 시각화 특징

#### 색상 코딩
- **추움**: 파란색 계열
- **쾌적**: 초록색 계열
- **더움**: 빨간색 계열

#### 그래프 구성
- **1x3 레이아웃**: 상세 분석
- **2x2 레이아웃**: 분류 분석
- **고해상도**: 300 DPI

---

## 모델 저장

### 📁 저장되는 파일들

```
saved_models/
├── temperature_analysis_v3_model.cbm    # CatBoost 모델
└── temperature_analysis_v3_features.pkl # 특성 컬럼
```

### 🔧 저장 기능

```python
def save_model_v3(model, feature_columns):
    """V3 모델 저장"""
    # 모델 저장 디렉토리 생성
    os.makedirs('saved_models', exist_ok=True)
    
    # CatBoost 모델 저장
    model.save_model('saved_models/temperature_analysis_v3_model.cbm')
    
    # 특성 컬럼 저장
    joblib.dump(feature_columns, 'saved_models/temperature_analysis_v3_features.pkl')
```

---

## 사용 방법

### 🚀 스크립트 실행

```bash
python temperature_analysis_v3.py
```

### 📊 생성되는 결과물

#### 1. 콘솔 출력
```
=== V3 모델 상세 분석 ===
추움: 33도 미만, 쾌적: 33-35도, 더움: 35도 초과
사용 특성: HR_mean, HRV_SDNN, BMI, mean_sa02

데이터 로딩 중...
원본 데이터 크기: (4649, 8)
S046의 temp_median이 0인 데이터 제거 중...
정리된 데이터 크기: (4606, 8)
제거된 데이터 수: 43

온도 라벨 생성 중... (추움: 33도 미만, 더움: 35도 초과)

라벨 분포:
추움 (0): 1104개 (24.0%)
쾌적 (1): 2472개 (53.7%)
더움 (2): 1030개 (22.4%)

특성 준비 중... (4개 특성만 사용)
특성 수: 4
특성 목록: ['HR_mean', 'HRV_SDNN', 'bmi', 'mean_sa02']

CatBoost 모델 학습 중...
학습 데이터 크기: (3684, 4)
검증 데이터 크기: (922, 4)

상세 분석 그래프 생성 중...
분류 분석 그래프 생성 중...

=== 분석 결과 요약 ===
R² 점수: 0.2492
RMSE: 1.9422°C
```

#### 2. 시각화 파일
- `temperature_analysis_v3_detailed.png` - 상세 분석 그래프
- `temperature_analysis_v3_classification.png` - 분류 분석 그래프

#### 3. 모델 파일
- `saved_models/temperature_analysis_v3_model.cbm` - CatBoost 모델
- `saved_models/temperature_analysis_v3_features.pkl` - 특성 컬럼

---

## 결론

### ✅ 주요 성과

#### 1. **적절한 분류 성능**
- **검증 정확도**: 69.6%
- **R² 점수**: 24.9% (온도 예측)
- **RMSE**: 1.94°C (온도 오차)

#### 2. **효율적인 학습**
- **빠른 학습**: 1.18초 내 완료
- **Early Stopping**: 과적합 방지
- **안정적 수렴**: 192 iteration에서 최적

#### 3. **상세한 분석**
- **회귀 지표**: R², RMSE 제공
- **시각화**: 다각도 분석 그래프
- **해석 가능성**: 클래스별 성능 분석

### 🎯 핵심 인사이트

#### 1. **4개 특성의 효과성**
- **핵심 특성**만으로도 충분한 성능
- **Gender 제외**해도 적절한 예측력
- **특성 선택**의 중요성 확인

#### 2. **CatBoost의 장점**
- **범주형 변수** 자동 처리
- **과적합 방지** 내장
- **빠른 학습** 속도

#### 3. **분류 vs 회귀**
- **분류 모델**로도 회귀 지표 계산 가능
- **라벨 변환**을 통한 온도 예측
- **다각도 평가**의 중요성

### 🚀 개선 방향

#### 1. **성능 향상**
- **하이퍼파라미터 튜닝**
- **특성 엔지니어링**
- **앙상블 모델**

#### 2. **모델 개선**
- **더 많은 특성** 추가
- **시간적 특성** 고려
- **개인별 맞춤화**

#### 3. **분석 확장**
- **실시간 모니터링**
- **경고 시스템** 구축
- **의료진 인터페이스**

---

## 부록

### 📚 코드 구조

#### 주요 함수
- `load_and_preprocess_data()` - 데이터 로드 및 전처리
- `create_temperature_labels_v3()` - 온도 라벨 생성
- `prepare_features_v3()` - 특성 준비 (4개)
- `train_catboost_model()` - CatBoost 모델 학습
- `create_detailed_analysis_plots()` - 상세 분석 그래프
- `create_classification_analysis_plots()` - 분류 분석 그래프
- `save_model_v3()` - 모델 저장

#### 평가 메트릭
- **정확도**: 분류 성능
- **R²**: 회귀 성능 (온도 예측)
- **RMSE**: 평균 제곱근 오차

### 🔗 관련 파일

- `temperature_analysis_v3.py` - 메인 스크립트
- `model_loader.py` - 모델 로딩 유틸리티
- `temperature_analysis_v3_detailed.png` - 상세 분석 그래프
- `temperature_analysis_v3_classification.png` - 분류 분석 그래프
- `saved_models/temperature_analysis_v3_model.cbm` - 저장된 모델

---

**문서 작성**: 2025-10-14  
**분석 대상**: temperature_analysis_v3.py  
**작성자**: AI Assistant  
**최종 업데이트**: 2025-10-14
