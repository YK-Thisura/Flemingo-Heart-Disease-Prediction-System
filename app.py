#pip install flask numpy joblib tensorflow

from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model('model/heart_disease_model.h5')
scaler = joblib.load('model/scaler.pkl')

# Input fields in correct order
features = ['age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form.get(feature, 0)) for feature in features]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        confidence = prediction[0][0]

        result = "💔 High risk of heart disease" if confidence > 0.5 else "❤️ Low risk of heart disease"
        return render_template('index.html', prediction=result, confidence=round(confidence, 2))
    except Exception as e:
        return render_template('index.html', prediction="Error occurred: " + str(e), confidence=None)

if __name__ == '__main__':
    app.run(debug=True)
