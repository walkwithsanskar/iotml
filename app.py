from flask import Flask, request, jsonify
import joblib
import requests

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load('naive_bayes_model2.pkl')

# Define label mapping
label_mapping = {
    1: 'fog',
    4: 'sun',
    0: 'drizzle',
    3: 'snow',
    2: 'rain'
}

# Constant API key
API_KEY = "cb5150ce2489ac77d9a01d2ba5c33c2d"

# Function to fetch weather data from OpenWeather API
def fetch_weather_data(city):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        data = response.json()
        precipitation = data['rain']['1h'] if 'rain' in data else 0
        temp_max = data['main']['temp_max']
        temp_min = data['main']['temp_min']
        wind = data['wind']['speed']
        return precipitation, temp_max, temp_min, wind
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    city = data.get('city')

    if not city:
        return jsonify({'error': 'City must be provided.'}), 400

    # Fetch weather data
    weather_data = fetch_weather_data(city)

    if weather_data is None:
        return jsonify({'error': 'Failed to fetch weather data. Check your city name and API key.'}), 400

    # Unpack weather data
    precipitation, temp_max, temp_min, wind = weather_data

    # Make prediction
    prediction = loaded_model.predict([[precipitation, temp_max, temp_min, wind]])

    # Map prediction back to original label
    predicted_weather = label_mapping[prediction[0]]

    # Return prediction
    return jsonify({'prediction': predicted_weather})
