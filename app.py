from flask import Flask, request, jsonify
from .classification_pipeline import load_model_and_predict

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    input_data = [data['vibration_level'], data['temperature_C'], data['pressure_PSI'], data['flow_rate_m3h']]
    prediction = load_model_and_predict(input_data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
