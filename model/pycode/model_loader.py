"""
간단한 모델 로딩 유틸리티
실제 저장된 V3+Gender 모델을 로드하고 예측하는 도구
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

class TemperaturePredictor:
    """체온 예측을 위한 간단한 클래스"""
    
    def __init__(self, model_path='temperature_prediction_model_v3_gender.cbm'):
        self.model_path = model_path
        self.model = None
        self.feature_columns = ['bmi', 'mean_sa02', 'HRV_SDNN', 'HR_mean', 'Gender_F', 'Gender_M']
        self.class_names = ['추움', '쾌적', '더움']
        
    def load_model(self):
        """모델 로드"""
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            print(f"모델 로드 완료: {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            return False
    
    def predict_temperature(self, data):
        """체온 예측"""
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 데이터 전처리
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # 필요한 특성만 선택
        X = data[self.feature_columns]
        
        # 예측
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # 결과 포맷팅
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'prediction': self.class_names[pred],
                'prediction_label': pred,
                'probabilities': {
                    '추움': prob[0],
                    '쾌적': prob[1], 
                    '더움': prob[2]
                },
                'confidence': max(prob)
            }
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """모델 정보 반환"""
        if self.model is None:
            return {"status": "모델이 로드되지 않음"}
        
        return {
            "status": "모델 로드됨",
            "model_path": self.model_path,
            "feature_columns": self.feature_columns,
            "class_names": self.class_names
        }

def example_usage():
    """사용 예시"""
    # 예측기 생성
    predictor = TemperaturePredictor()
    
    # 모델 로드
    if predictor.load_model():
        # 예측을 위한 샘플 데이터
        sample_data = {
            'bmi': 22.5,
            'mean_sa02': 98.5,
            'HRV_SDNN': 45.2,
            'HR_mean': 72.3,
            'Gender_F': 1,  # 여성
            'Gender_M': 0   # 남성
        }
        
        # 예측 수행
        results = predictor.predict_temperature(sample_data)
        
        print("=== 체온 예측 결과 ===")
        for i, result in enumerate(results):
            print(f"샘플 {i+1}:")
            print(f"  예측: {result['prediction']}")
            print(f"  신뢰도: {result['confidence']:.3f}")
            print(f"  확률 분포:")
            for class_name, prob in result['probabilities'].items():
                print(f"    {class_name}: {prob:.3f}")

if __name__ == "__main__":
    example_usage()
