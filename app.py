from flask import Flask, request, jsonify
import pickle
import numpy as np
import datetime

# Load trained Random Forest model
with open("notebooks/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load label encoders for categorical features
with open("notebooks/crime_type_encoder.pkl", "rb") as pt_file:
    primary_type_encoder = pickle.load(pt_file)

with open("notebooks/location_encoder.pkl", "rb") as ld_file:
    location_encoder = pickle.load(ld_file)

app = Flask(__name__)

@app.route("/")
def home():
    return "Crime Prediction API is running "

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        # Extract features from input
        year = data["year"]
        month = data["month"]
        day = data["day"]
        hour = data["hour"]  # Ensure you pass this from frontend
        weekday = datetime.datetime(year, month, day).weekday()  # Get weekday (0=Monday, 6=Sunday)
        is_night = 1 if (hour < 6 or hour > 18) else 0  # Example logic for night hours

        location_desc = location_encoder.transform([data["location"]])[0]  # Encode location
        arrest = data["arrest"]  # Boolean value
        domestic = data["domestic"]  # Boolean value
        beat = data["beat"]
        district = data["district"]
        ward = data["ward"]
        community_area = data["community_area"]
        latitude = data["latitude"]
        longitude = data["longitude"]

        # Ensure input shape matches the model's expected input
        features = np.array([
            hour, weekday, is_night, location_desc, arrest, domestic, 
            beat, district, ward, community_area, latitude, longitude
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert prediction back to crime type
        predicted_crime = primary_type_encoder.inverse_transform([prediction])[0]

        return jsonify({"predicted_crime_type": predicted_crime})

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400

if __name__ == '__main__':
    app.run(debug=True)