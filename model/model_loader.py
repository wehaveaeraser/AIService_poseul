"""
모델 로딩 유틸리티
저장된 모델들을 쉽게 로드하고 예측할 수 있는 도구
"""

import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

class ModelLoader:
    """모델 로딩 및 예측을 위한 클래스"""
    
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.feature_columns = {}
        
    def load_multiple_classifiers(self):
        """다중 분류기 모델들 로드"""
        print("다중 분류기 모델들 로드 중...")
        
        # 스케일러 로드
        self.scaler = joblib.load(f'{self.model_dir}/scaler.pkl')
        print("스케일러 로드 완료")
        
        # 특성 컬럼 로드
        self.feature_columns['multiple'] = joblib.load(f'{self.model_dir}/feature_columns.pkl')
        print("특성 컬럼 로드 완료")
        
        # 각 모델 로드
        model_files = [
            'random_forest_model.pkl',
            'xgboost_model.pkl', 
            'svm_model.pkl',
            'logistic_regression_model.pkl',
            'k_nearest_neighbors_model.pkl',
            'naive_bayes_model.pkl',
            'decision_tree_model.pkl'
        ]
        
        for model_file in model_files:
            try:
                model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                self.models[model_name] = joblib.load(f'{self.model_dir}/{model_file}')
                print(f"{model_name} 모델 로드 완료")
            except FileNotFoundError:
                print(f"{model_file} 파일을 찾을 수 없습니다.")
        
        # CatBoost 모델 로드
        try:
            catboost_model = CatBoostClassifier()
            catboost_model.load_model(f'{self.model_dir}/catboost_model.cbm')
            self.models['CatBoost'] = catboost_model
            print("CatBoost 모델 로드 완료")
        except FileNotFoundError:
            print("CatBoost 모델 파일을 찾을 수 없습니다.")
    
    def load_v3_model(self):
        """V3 모델 로드"""
        print("V3 모델 로드 중...")
        
        # CatBoost 모델 로드
        self.models['V3'] = CatBoostClassifier()
        self.models['V3'].load_model(f'{self.model_dir}/temperature_analysis_v3_model.cbm')
        
        # 특성 컬럼 로드
        self.feature_columns['V3'] = joblib.load(f'{self.model_dir}/temperature_analysis_v3_features.pkl')
        
        print("V3 모델 로드 완료")
    
    def load_v3_gender_model(self):
        """V3 Gender 모델 로드"""
        print("V3 Gender 모델 로드 중...")
        
        # CatBoost 모델 로드
        self.models['V3_Gender'] = CatBoostClassifier()
        self.models['V3_Gender'].load_model(f'{self.model_dir}/temperature_prediction_model_v3_gender.cbm')
        
        # 특성 컬럼 로드
        self.feature_columns['V3_Gender'] = joblib.load(f'{self.model_dir}/temperature_prediction_v3_gender_features.pkl')
        
        # 특성 중요도 로드
        self.feature_importance = joblib.load(f'{self.model_dir}/temperature_prediction_v3_gender_importance.pkl')
        
        print("V3 Gender 모델 로드 완료")
    
    def predict_temperature(self, data, model_name='CatBoost'):
        """체온 예측"""
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이 로드되지 않았습니다.")
        
        # 데이터 전처리
        if model_name in ['multiple', 'V3_Gender']:
            # 6개 특성 모델
            features = ['bmi', 'mean_sa02', 'HRV_SDNN', 'HR_mean', 'Gender_F', 'Gender_M']
        else:
            # 4개 특성 모델
            features = ['HR_mean', 'HRV_SDNN', 'bmi', 'mean_sa02']
        
        # 특성 선택
        X = data[features]
        
        # 스케일링 (SVM, Logistic Regression, KNN에만 필요)
        if model_name in ['SVM', 'Logistic Regression', 'K-Nearest Neighbors']:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # 예측
        predictions = self.models[model_name].predict(X_scaled)
        probabilities = self.models[model_name].predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def get_model_info(self):
        """로드된 모델 정보 반환"""
        info = {
            'loaded_models': list(self.models.keys()),
            'feature_columns': self.feature_columns,
            'scaler_loaded': self.scaler is not None
        }
        return info

def example_usage():
    """사용 예시"""
    # 모델 로더 생성
    loader = ModelLoader()
    
    # 모델들 로드
    loader.load_multiple_classifiers()
    loader.load_v3_model()
    loader.load_v3_gender_model()
    
    # 모델 정보 확인
    print(loader.get_model_info())
    
    # 예측을 위한 샘플 데이터 생성
    sample_data = pd.DataFrame({
        'bmi': [22.5],
        'mean_sa02': [98.5],
        'HRV_SDNN': [45.2],
        'HR_mean': [72.3],
        'Gender_F': [1],
        'Gender_M': [0]
    })
    
    # 예측 수행
    predictions, probabilities = loader.predict_temperature(sample_data, 'CatBoost')
    print(f"예측 결과: {predictions}")
    print(f"예측 확률: {probabilities}")

if __name__ == "__main__":
    example_usage()
