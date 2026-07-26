from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and transformers
best_gbr = joblib.load('best_gbr_model.pkl')
label_encoder_location = joblib.load('label_encoder_location.pkl')
label_encoder_society = joblib.load('label_encoder_society.pkl')
#poly = joblib.load('poly_transformer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input values from the form
        location = request.form['location']
        society = request.form['society']
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])
        balcony = int(request.form['balcony'])
        total_sqft = float(request.form['total_sqft'])

        # Preprocess the input values
        # Ensure location and society are label encoded in the same way as during training
        location_encoded = label_encoder_location.transform([location])[0]
        society_encoded = label_encoder_society.transform([society])[0]

        # Create a DataFrame with the input values in the same order as during training
        input_data = pd.DataFrame([[location_encoded, society_encoded, total_sqft,bath,balcony,bhk  ]], 
                                  columns=['location', 'society', 'total_sqft','bath','balcony','bhk' ])

        # Apply Polynomial Transformation (same as training)
        #input_data_poly = poly.transform(input_data)

        # Predict the price using the trained model
        predicted_price = best_gbr.predict(input_data)

        # Return the predicted price to the user
        return render_template('index.html', predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
