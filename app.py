from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
try:
    regmodel = pickle.load(open('regmodel.pkl', 'rb'))
    scaler = pickle.load(open('scaling.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    exit(1)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print("Received data:", data)

        # Handle list of dictionaries or a single dictionary
        if isinstance(data, list):
            data = data[0]  # Process the first item if it's a list

        # Convert input data to a NumPy array
        input_array = np.array(list(data.values())).reshape(1, -1)
        print("Input array:", input_array)

        # Apply scaling
        new_data = scaler.transform(input_array)
        print("Scaled data:", new_data)

        # Make prediction
        output = regmodel.predict(new_data)
        print("Prediction:", output[0])

        return jsonify({'prediction': output[0]})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
