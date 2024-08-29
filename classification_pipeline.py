import numpy as np
import joblib


def load_model_and_predict(input_data):
    model = joblib.load('models/pump_failure_rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    input_data = np.array(input_data).reshape(1, -1)
    
    scaled_data = scaler.transform(input_data)
    
    prediction = model.predict(scaled_data)
    prediction_prob = model.predict_proba(scaled_data)
    
    if prediction == 1:
        return f"Pump failure is likely. Prediction Probability: {prediction_prob[0][1]:.2f}"
    else:
        return f"Pump is operating normally. Prediction Probability: {prediction_prob[0][0]:.2f}"

def get_user_input():
    vibration_level = float(input("Enter vibration level: "))
    temperature_C = float(input("Enter temperature in Celsius: "))
    pressure_PSI = float(input("Enter pressure in PSI: "))
    flow_rate_m3h = float(input("Enter flow rate in m3/h: "))
    
    return [vibration_level, temperature_C, pressure_PSI, flow_rate_m3h]

user_input = get_user_input()

result = load_model_and_predict(user_input)
print(result)

