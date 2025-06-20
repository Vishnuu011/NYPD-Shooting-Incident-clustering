from server.cluster_app import PredictPipeline
from flask import (
    Flask,
    request,
    jsonify
)
import sys
import sys
import os

def create_app():
    app = Flask(__name__)
    predictor = PredictPipeline()

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.json
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            cluster, prob = predictor.predict_cluster(lat, lon)
            return jsonify({
                'cluster': cluster,
                'probability': prob,
                'latitude': lat,
                'longitude': lon
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/')
    def index():
        return "NYPD Shooting Cluster Prediction API - POST /predict with {'latitude': <value>, 'longitude': <value>}"

    return app

if __name__ == '__main__':
    app = create_app()
    print("Starting Flask API server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)