# Temperature Prediction Models

## 🐍 Python 모델 파일들

| 파일명 | 목적 | 알고리즘 | 특성 | 데이터 | 성능 | 출력 |
|--------|------|----------|------|--------|------|------|
| `temperature_prediction_model_v3_gender.py` | 온도 예측 모델 (성별 특성 포함) | CatBoost Classifier | 6개 (BMI, mean_sa02, HRV_SDNN, HR_mean, Gender_F, Gender_M) | 4,606개 샘플 (S046 이상치 제거 후) | 71.15% 정확도 | - `temperature_prediction_model_v3_gender.cbm` (학습된 모델) <br> - `temperature_prediction_results_v3_gender.png` (기본 결과 그래프) <br> - `temperature_analysis_v3_gender_detailed.png` (상세 분석 그래프) |
| `multiple_classifiers_comparison.py` | 8가지 분류 알고리즘 성능 비교 | Random Forest, XGBoost, SVM, Logistic Regression, KNN, Naive Bayes, Decision Tree, CatBoost | 6개 | 4,606개 샘플 | CatBoost F1-Score 0.6866 | `multiple_classifiers_comparison_f1.png` (종합 비교 그래프) |
| `temperature_analysis_v3.py` | V3 모델 상세 분석 (분류 + 회귀 성능) | CatBoost Classifier | 4개 (HR_mean, HRV_SDNN, BMI, mean_sa02) - Gender 제외 | 4,606개 샘플 | 분류 69.6% 정확도 <br> 회귀 R² 24.9%, RMSE 1.94°C | - `temperature_analysis_v3_detailed.png` (상세 분석 그래프) <br> - `temperature_analysis_v3_classification.png` (분류 분석 그래프) |

---

## 📊 데이터 파일들
https://physionet.org/content/dreamt/2.1.0/data_64Hz/#files-panel

원데이터 https://github.com/WillKeWang/DREAMT_FE

DREAMT: 멀티센서 웨어러블 기술을 이용한 실시간 수면 단계 추정 데이터셋

## 정제 데이터
| 파일명 | 용도 | 크기/특성 |
|--------|------|-----------|
| `extracted_data_sampled_20rows.csv` | 실시간 생체신호 데이터 (샘플) | 20행 (실험용) |
| `extracted_data_with_sid_info.csv` | 실시간 생체신호 데이터 (전체) | 4,649행 → 4,606행 (S046 이상치 43개 제거 후) <br> 특성: BMI, mean_sa02, HRV_SDNN, HR_mean, gender, TEMP_median 등 |

---

## 📈 결과 파일들

| 파일명 | 내용 | 구성 |
|--------|------|------|
| `multiple_classifiers_comparison_f1.png` | 8개 모델의 F1-Score 비교 그래프 | 4개 서브플롯 (F1-Score vs 정확도, 클래스별 성능, 특성 중요도, 순위) |
| `temperature_prediction_results_v3_gender.png` | V3+Gender 모델 기본 결과 | 혼동 행렬 + 특성 중요도 |
| `temperature_analysis_v3_gender_detailed.png` | V3+Gender 모델 상세 분석 | 4개 서브플롯 (혼동 행렬, 클래스별 정확도, 특성 중요도, 예측 확률 분포) |

---

## 📋 보고서 파일들

| 파일명 | 내용 | 성능/결과 |
|--------|------|-----------|
| `Temperature_Prediction_Model_V3_Gender_Report.md` | V3+Gender 모델 상세 보고서 | 71.15% 정확도, BMI 34.03% 중요도 <br> 포함: 성능 분석, 특성 중요도, 시각화 결과, 사용 방법 |
| `Multiple_Classifiers_Comparison_Report.md` | 8가지 분류기 비교 보고서 | CatBoost 1위 (F1-Score 0.6866) <br> 포함: 알고리즘별 성능, 클래스별 분석, 특성 중요도 비교 |
| `Temperature_Analysis_V3_Report.md` | V3 모델 종합 분석 보고서 | 분류 69.6% 정확도, 회귀 R² 24.9% <br> 포함: 분류+회귀 성능, 상세 시각화, 실용성 검증 |

---

## 🎯 모델별 핵심 특징 (실제 데이터 기준)

| 모델 | 파일 | 특성 수 | 데이터 | 정확도 | 주요 특징 |
|------|------|--------|--------|--------|-----------|
| V3+Gender | `temperature_prediction_model_v3_gender.py` | 6개 | 4,606개 | 71.15% | 성별 특성 포함, 최고 성능 |
| V3 | `temperature_analysis_v3.py` | 4개 | 4,606개 | 69.6% | 분류+회귀 이중 평가 |
| 비교 | `multiple_classifiers_comparison.py` | 6개 | 4,606개 | 68.66% | 8가지 알고리즘 성능 비교 |

---

## 📊 클래스 분포 (실제 데이터)

| 클래스 | 온도 범위 | 샘플 수 | 비율 |
|--------|-----------|---------|------|
| 추움 | 33°C 미만 | 1,104개 | 24.0% |
| 쾌적 | 33–35°C | 2,472개 | 53.7% |
| 더움 | 35°C 초과 | 1,030개 | 22.4% |

---

## 🔍 특성 중요도 (V3+Gender 모델 기준)

| 순위 | 특성 | 중요도 | 설명 |
|------|------|--------|------|
| 1 | BMI | 34.03% | 체질량지수가 가장 중요 |
| 2 | mean_sa02 | 27.16% | 평균 산소포화도 |
| 3 | HRV_SDNN | 13.80% | 심박변이도 |
| 4 | Gender_F | 9.85% | 여성 특성 ⭐ |
| 5 | HR_mean | 8.79% | 평균 심박수 |
| 6 | Gender_M | 6.36% | 남성 특성 ⭐ |
