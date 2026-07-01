from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("models/loan_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    data = request.form

    input_data = [
        float(data['gender']),
        float(data['married']),
        float(data['dependents']),
        float(data['education']),
        float(data['self_employed']),
        float(data['applicant_income']),
        float(data['coapplicant_income']),
        float(data['loan_amount']),
        float(data['loan_term']),
        float(data['credit_history']),
        float(data['property_area'])
    ]

    final_input = np.array(input_data).reshape(1, -1)

    prediction = model.predict(final_input)

    result = "Congratulations Loan Approved." if prediction[0] == 1 else "Loan Rejected"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
