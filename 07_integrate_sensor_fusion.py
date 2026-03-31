"""
EWS Integration System
Menggabungkan Sensor Ultrasonik + Computer Vision dengan Decision Fusion
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple


class SensorModel:
    """Load dan use trained Random Forest sensor model with fallback"""
    
    def __init__(self, model_path, le_status_path, le_weather_path):
        """
        Initialize sensor model with trained Random Forest
        Falls back to heuristic model if pickle fails
        
        Args:
            model_path: Path ke rf_ews_model.pkl
            le_status_path: Path ke le_status.pkl (status encoder)
            le_weather_path: Path ke le_weather.pkl (weather encoder)
        """
        self.model = None
        self.le_status = None
        self.le_weather = None
        self.use_heuristic = False
        
        try:
            # Try to load RF model with pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load label encoders
            with open(le_status_path, 'rb') as f:
                self.le_status = pickle.load(f)
            
            with open(le_weather_path, 'rb') as f:
                self.le_weather = pickle.load(f)
            
            print("✓ Sensor model loaded successfully")
            print(f"  Model type: {type(self.model).__name__}")
            print(f"  Status classes: {list(self.le_status.classes_)}")
        
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            print(f"⚠️  Warning: Could not load pickle model: {e}")
            print("  Falling back to heuristic-based sensor model")
            self.use_heuristic = True
            
            # Define fallback label encoders
            from sklearn.preprocessing import LabelEncoder
            self.le_status = LabelEncoder()
            self.le_status.classes_ = np.array(['Aman', 'Siaga', 'Waspada', 'Bahaya'])
            
            self.le_weather = LabelEncoder()
            self.le_weather.classes_ = np.array(['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Sedang', 'Hujan Lebat'])
    
    def predict(self, water_level_cm: float, rainfall_mm: float, weather: str) -> Tuple[str, float]:
        """
        Predict flood status dari sensor data
        
        Args:
            water_level_cm: Ketinggian air dalam cm (dari ultrasonik)
            rainfall_mm: Curah hujan dalam mm
            weather: Kondisi cuaca (string)
        
        Returns:
            status: Flood status (Aman/Siaga/Waspada/Bahaya)
            confidence: Confidence score (0-1)
        """
        if self.use_heuristic:
            # Use heuristic model if RF failed to load
            return self._heuristic_predict(water_level_cm, rainfall_mm, weather)
        
        # Use trained RF model
        try:
            # Encode weather
            weather_encoded = self.le_weather.transform([weather])[0]
            
            # Prepare features
            features = np.array([[water_level_cm, rainfall_mm, weather_encoded]])
            
            # Predict
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities.max()
            
            # Decode status
            status = self.le_status.inverse_transform([prediction])[0]
            
            return status, float(confidence)
        
        except Exception as e:
            print(f"⚠️  RF prediction failed: {e}, using heuristic")
            return self._heuristic_predict(water_level_cm, rainfall_mm, weather)
    
    def _heuristic_predict(self, water_level_cm: float, rainfall_mm: float, weather: str) -> Tuple[str, float]:
        """
        Heuristic-based prediction based on water level and rainfall
        
        Status mapping:
        - Aman: < 50 cm
        - Siaga: 50-100 cm
        - Waspada: 100-150 cm
        - Bahaya: > 150 cm
        
        Heavy rainfall increases confidence
        """
        # Base status on water level
        if water_level_cm < 50:
            status = 'Aman'
            base_conf = 0.95
        elif water_level_cm < 100:
            status = 'Siaga'
            base_conf = 0.85
        elif water_level_cm < 150:
            status = 'Waspada'
            base_conf = 0.80
        else:
            status = 'Bahaya'
            base_conf = 0.90
        
        # Adjust confidence based on rainfall
        rainfall_factor = min(rainfall_mm / 100.0, 1.0)  # Scale 0-1
        confidence = base_conf * (0.7 + 0.3 * rainfall_factor)  # 70-100% confidence range
        
        return status, float(confidence)


class DecisionFusion:
    """
    Menggabungkan prediksi sensor + CV dengan AND logic
    Hanya trigger alarm jika KEDUA sumber setuju
    """
    
    @staticmethod
    def status_to_level(status: str) -> int:
        """Convert status string to numeric level"""
        levels = {
            'Aman': 0,
            'Siaga': 1,
            'Waspada': 2,
            'Bahaya': 3
        }
        return levels.get(status, 0)
    
    @staticmethod
    def level_to_status(level: int) -> str:
        """Convert numeric level to status string"""
        statuses = {
            0: 'Aman',
            1: 'Siaga',
            2: 'Waspada',
            3: 'Bahaya'
        }
        return statuses.get(level, 'Aman')
    
    @classmethod
    def fuse_predictions(cls, 
                        sensor_status: str, 
                        sensor_confidence: float,
                        cv_status: str, 
                        cv_confidence: float) -> Dict:
        """
        Fusion dua prediksi dengan AND logic
        
        Args:
            sensor_status: Status dari sensor (Aman/Siaga/Waspada/Bahaya)
            sensor_confidence: Confidence dari sensor (0-1)
            cv_status: Status dari CV model
            cv_confidence: Confidence dari CV model
        
        Returns:
            Fused decision dengan reasoning
        """
        sensor_level = cls.status_to_level(sensor_status)
        cv_level = cls.status_to_level(cv_status)
        
        # AND Logic: Ambil level LEBIH TINGGI jika kedua sources sepakat signifikan
        fused_level = sensor_level
        agreement = abs(sensor_level - cv_level) <= 1  # Maksimal 1 level difference
        
        if not agreement:
            # Jika sources disagree, ambil level lebih tinggi dengan confidence penalty
            fused_level = max(sensor_level, cv_level)
            combined_confidence = min(sensor_confidence, cv_confidence) * 0.7
        else:
            # Jika sources sepakat, kombinasi confidence
            combined_confidence = (sensor_confidence + cv_confidence) / 2
        
        fused_status = cls.level_to_status(fused_level)
        
        # Determine decision
        if fused_level >= 2 and agreement:
            # Both sources indicate danger AND they agree
            decision = 'TRIGGER_ALARM'
            recommendation = f"HIGH FLOOD RISK - Both sensor ({sensor_status}) and camera ({cv_status}) confirm!"
        elif fused_level >= 2 and not agreement:
            # One source indicates danger but disagreement
            decision = 'VERIFY_ANOMALY'
            recommendation = f"Disagreement detected: Sensor={sensor_status}, Camera={cv_status}. Verify data quality."
        elif fused_level >= 1:
            # Alert level
            decision = 'MONITOR'
            recommendation = f"Status {fused_status}: Monitor water levels and weather conditions."
        else:
            # Safe
            decision = 'NO_ALARM'
            recommendation = "Conditions normal. No flood risk detected."
        
        return {
            'fused_status': fused_status,
            'fused_level': fused_level,
            'confidence': combined_confidence,
            'agreement': agreement,
            'decision': decision,
            'recommendation': recommendation,
            'reasoning': {
                'sensor': {
                    'status': sensor_status,
                    'confidence': sensor_confidence,
                    'level': sensor_level
                },
                'cv': {
                    'status': cv_status,
                    'confidence': cv_confidence,
                    'level': cv_level
                },
                'fusion_method': 'AND Logic - Both sources must agree on danger level'
            }
        }


class EWSIntegration:
    """
    Integrated Early Warning System
    Menggabungkan Sensor + CV models dengan decision fusion
    """
    
    def __init__(self, 
                 sensor_model_path: str,
                 le_status_path: str,
                 le_weather_path: str,
                 cv_model_path: str = None,
                 device: str = 'cuda'):
        """
        Initialize integrated EWS system
        
        Args:
            sensor_model_path: Path ke rf_ews_model.pkl
            le_status_path: Path ke le_status.pkl
            le_weather_path: Path ke le_weather.pkl
            cv_model_path: Path ke best_model.pth (optional, can be added later)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Load sensor model
        try:
            self.sensor = SensorModel(sensor_model_path, le_status_path, le_weather_path)
            print("✓ Sensor model initialized")
        except Exception as e:
            print(f"✗ Failed to load sensor model: {e}")
            raise
        
        # Load CV model if available
        self.cv = None
        if cv_model_path and Path(cv_model_path).exists():
            try:
                from _06_inference import FloodDetector
                self.cv = FloodDetector(cv_model_path, device=device)
                print("✓ CV model initialized")
            except Exception as e:
                print(f"⚠️  Warning: CV model not loaded: {e}")
                print("   Continuing with sensor-only mode")
    
    def predict(self, 
                water_level_cm: float,
                rainfall_mm: float,
                weather: str,
                image_path: str = None) -> Dict:
        """
        Make prediction using both sensor and CV (if available)
        
        Args:
            water_level_cm: Water level from ultrasonic sensor (cm)
            rainfall_mm: Rainfall amount (mm)
            weather: Weather condition (string)
            image_path: Path to image for CV model (optional)
        
        Returns:
            Integrated prediction with decision
        """
        timestamp = datetime.now().isoformat()
        
        # Get sensor prediction
        sensor_status, sensor_conf = self.sensor.predict(water_level_cm, rainfall_mm, weather)
        
        result = {
            'timestamp': timestamp,
            'sensor': {
                'water_level_cm': water_level_cm,
                'rainfall_mm': rainfall_mm,
                'weather': weather,
                'status': sensor_status,
                'confidence': sensor_conf
            },
            'cv': None,
            'fusion': None
        }
        
        # Get CV prediction if available
        if self.cv and image_path:
            try:
                cv_result = self.cv.process_image(image_path)
                cv_status = cv_result['flood_status']
                cv_conf = 1.0 if cv_result['flood_detected'] else 0.5
                
                result['cv'] = {
                    'image': image_path,
                    'water_percentage': cv_result['water_percentage'],
                    'status': cv_status,
                    'confidence': cv_conf,
                    'flood_detected': cv_result['flood_detected']
                }
                
                # Fuse predictions
                fusion_result = DecisionFusion.fuse_predictions(
                    sensor_status, sensor_conf,
                    cv_status, cv_conf
                )
                result['fusion'] = fusion_result
            
            except Exception as e:
                print(f"⚠️  CV prediction failed: {e}")
                result['cv'] = {'error': str(e)}
                # Use sensor-only decision
                result['fusion'] = self._sensor_only_decision(sensor_status, sensor_conf)
        else:
            # Sensor-only mode if CV not available
            result['fusion'] = self._sensor_only_decision(sensor_status, sensor_conf)
        
        return result
    
    @staticmethod
    def _sensor_only_decision(status: str, confidence: float) -> Dict:
        """Generate decision based on sensor only"""
        levels = {'Aman': 0, 'Siaga': 1, 'Waspada': 2, 'Bahaya': 3}
        level = levels.get(status, 0)
        
        if level >= 2:
            decision = 'TRIGGER_ALARM'
            recommendation = f"Sensor indicates {status} - Action required!"
        elif level >= 1:
            decision = 'MONITOR'
            recommendation = f"Status {status} - Monitor conditions."
        else:
            decision = 'NO_ALARM'
            recommendation = "Conditions normal."
        
        return {
            'fused_status': status,
            'confidence': confidence,
            'decision': decision,
            'recommendation': recommendation,
            'mode': 'sensor_only',
            'note': 'CV model not available - using sensor data only'
        }
    
    def save_result_json(self, result: Dict, output_path: str):
        """Save prediction result to JSON"""
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Result saved to {output_path}")


def main():
    """Test EWS integration"""
    
    print("="*80)
    print("FLOOD EWS INTEGRATION TEST")
    print("="*80)
    
    # Paths
    sensor_model_path = Path("River-EWS/models/flood_dataset/rf_ews_model.pkl")
    le_status_path = Path("River-EWS/models/flood_dataset/le_status.pkl")
    le_weather_path = Path("River-EWS/models/flood_dataset/le_weather.pkl")
    cv_model_path = Path("checkpoints/best_model.pth")
    
    # Check files exist
    if not sensor_model_path.exists():
        print(f"❌ Sensor model not found: {sensor_model_path}")
        return
    
    try:
        # Initialize EWS
        print("\n[1] INITIALIZING EWS SYSTEM")
        ews = EWSIntegration(
            str(sensor_model_path),
            str(le_status_path),
            str(le_weather_path),
            str(cv_model_path) if cv_model_path.exists() else None,
            device='cuda'
        )
        
        # Test case 1: Low water level (normal condition)
        print("\n[2] TEST CASE 1: Normal Conditions")
        result1 = ews.predict(
            water_level_cm=30.0,
            rainfall_mm=5.0,
            weather="Cerah",
            image_path=None
        )
        print(f"Status: {result1['fusion']['fused_status']}")
        print(f"Decision: {result1['fusion']['decision']}")
        print(f"Recommendation: {result1['fusion']['recommendation']}")
        
        # Test case 2: Elevated water level
        print("\n[3] TEST CASE 2: Elevated Water Level")
        result2 = ews.predict(
            water_level_cm=120.0,
            rainfall_mm=50.0,
            weather="Hujan",
            image_path=None
        )
        print(f"Status: {result2['fusion']['fused_status']}")
        print(f"Decision: {result2['fusion']['decision']}")
        print(f"Recommendation: {result2['fusion']['recommendation']}")
        
        # Test case 3: Danger level
        print("\n[4] TEST CASE 3: Danger Level")
        result3 = ews.predict(
            water_level_cm=160.0,
            rainfall_mm=100.0,
            weather="Hujan Deras",
            image_path=None
        )
        print(f"Status: {result3['fusion']['fused_status']}")
        print(f"Decision: {result3['fusion']['decision']}")
        print(f"Recommendation: {result3['fusion']['recommendation']}")
        
        # Save results
        output_dir = Path("ews_results")
        output_dir.mkdir(exist_ok=True)
        ews.save_result_json(result1, str(output_dir / "test_case_1_normal.json"))
        ews.save_result_json(result2, str(output_dir / "test_case_2_elevated.json"))
        ews.save_result_json(result3, str(output_dir / "test_case_3_danger.json"))
        
        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)
        print(f"✓ Results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
