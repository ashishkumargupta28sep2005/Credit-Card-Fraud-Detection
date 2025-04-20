from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
xgb_model = joblib.load('xgb_model_final.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "✅ Welcome to the Credit Card Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = xgb_model.predict(features_scaled)
        fraud_prob = xgb_model.predict_proba(features_scaled)[0][1]

        return jsonify({
            'prediction': 'Fraud' if prediction[0] == 1 else 'Not Fraud',
            'fraud_probability': float(round(fraud_prob, 4))
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

