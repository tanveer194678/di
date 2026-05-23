import joblib
import numpy as np
from flask import Flask, request, jsonify

# Make sure this.pkl was trained on the diabetes dataset
model = joblib.load('diabetes_pipeline.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Diabetes Prediction using Flask API</h1>'

@app.route('/predict', methods=['POST']) # 'POST' must be uppercase
def predict():
    data = request.json
    
    # Pima Indians Diabetes dataset features
    features = np.array([[
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]])
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    result = 'Diabetic'
    if prediction[0] == 0:
        result = 'Not Diabetic'

    confidence = round(np.max(probability) * 100, 2)
    
    return jsonify({
        'prediction': result,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )