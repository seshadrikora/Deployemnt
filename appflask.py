from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model, encoder, and scaler
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Renders a template with a form for input

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        num1 = float(request.form['SL'])
        num2 = float(request.form['SW'])
        num3 = float(request.form['PL'])
        num4 = float(request.form['PW'])

        # Create an array of the input features
        features = np.array([[num1, num2, num3, num4]])

        # Scale the features
        scaled_features = scaler.transform(features)

        # Make the prediction
        prediction = model.predict(scaled_features)

        # Decode the predicted class
        final_predictions = encoder.inverse_transform(list(prediction))

        return render_template('result.html', prediction=final_predictions[0])  # Pass the prediction to a results page

if __name__ == '__main__':
    app.run(debug=True)

