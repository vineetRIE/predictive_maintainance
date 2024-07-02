from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained models and scaler
model_class = joblib.load('predictive_model_class.pkl')
model_reg = joblib.load('predictive_model_reg.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])
        
        # Convert input data to float
        input_df = input_df.astype(float)
        
        # Preprocess input data using the saved scaler
        input_processed = scaler.transform(input_df)
        
        # Predict maintenance needs
        prediction_class = model_class.predict(input_processed)
        
        # Predict expected days to maintenance
        prediction_reg = model_reg.predict(input_processed)
        
        # Generate a user-friendly message and expected days
        if prediction_class[0] == 1:
            message = "Yes, the machine needs preventive maintenance."
        else:
            message = "No need for preventive maintenance now."
        
        expected_days = f"Expected days to maintenance: {int(prediction_reg[0])} days"
        
        return redirect(url_for('result', prediction=message, expected_days=expected_days))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    expected_days = request.args.get('expected_days')
    return render_template('result.html', prediction=prediction, expected_days=expected_days)

if __name__ == '__main__':
    app.run(debug=True)
