import joblib
import numpy as np

model_file = "test_model.pkl"

def test_model_prediction():
    try:
        model = joblib.load(model_file)
        test_features = np.random.rand(1, 20)  # Replace with actual feature dimensions
        prediction = model.predict(test_features)
        print("Prediction:", prediction)
    except Exception as e:
        print(f"Model loading or prediction failed: {e}")

test_model_prediction()