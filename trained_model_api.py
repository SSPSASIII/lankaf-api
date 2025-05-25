#!/usr/bin/env python3
"""
LankaForecaster Trained Model API
=================================
Professional Flask API that matches your successfully working setup.
This uses joblib for model loading (as identified in your diagnostic).

Author: Professional Setup Assistant
Date: 2025-05-25
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LankaForecasterAPI:
    """Professional API class for LankaForecaster predictions"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Model storage
        self.revenue_model = None
        self.profit_model = None
        self.scaler = None
        self.model_info = None
        self.feature_columns = None
        
        # Load models
        self.load_models()
        
        # Setup routes
        self.setup_routes()
    
    def load_models(self):
        """Load models using joblib (your working method)"""
        try:
            # Model file paths
            model_files = {
                'revenue': 'lankaforecaster_revenue_model.pkl',
                'profit': 'lankaforecaster_profit_model.pkl',
                'scaler': 'lankaforecaster_scaler.pkl',
                'info': 'lankaforecaster_model_info.json'
            }
            
            # Check if files exist
            missing_files = []
            for name, filename in model_files.items():
                if not os.path.exists(filename):
                    missing_files.append(filename)
            
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {missing_files}")
            
            # Load models with joblib (your working method)
            print("🔄 Loading models with joblib...")
            self.revenue_model = joblib.load(model_files['revenue'])
            self.profit_model = joblib.load(model_files['profit'])
            self.scaler = joblib.load(model_files['scaler'])
            
            # Load model info
            with open(model_files['info'], 'r') as f:
                self.model_info = json.load(f)
            
            self.feature_columns = self.model_info.get('feature_columns', [])
            
            print("✅ All models loaded successfully with joblib!")
            print(f"📊 Revenue Model Accuracy: {self.model_info.get('revenue_accuracy', 'N/A')}")
            print(f"💰 Profit Model Accuracy: {self.model_info.get('profit_accuracy', 'N/A')}")
            print(f"🎯 Overall Accuracy: {self.model_info.get('overall_accuracy', 'N/A')}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            print(f"❌ Error loading models: {e}")
            print("Make sure model files are in the current directory")
            
            # Set models to None to indicate failure
            self.revenue_model = None
            self.profit_model = None
            self.scaler = None
            self.model_info = {'error': str(e)}
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            status = "healthy" if self.models_loaded() else "unhealthy"
            return jsonify({
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'models_loaded': self.models_loaded(),
                'api_version': '1.0.0'
            })
        
        @self.app.route('/model-details', methods=['GET'])
        def model_details():
            """Get model details and accuracy information"""
            if not self.models_loaded():
                return jsonify({
                    'error': 'Models not loaded',
                    'details': self.model_info
                }), 500
            
            return jsonify({
                'model_info': self.model_info,
                'feature_columns': self.feature_columns,
                'models_status': {
                    'revenue_model': 'loaded',
                    'profit_model': 'loaded',
                    'scaler': 'loaded'
                }
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Make predictions using the trained models"""
            if not self.models_loaded():
                return jsonify({
                    'error': 'Models not loaded',
                    'message': 'Please check model files and restart the API'
                }), 500
            
            try:
                # Get data from request
                data = request.get_json()
                
                if not data:
                    return jsonify({
                        'error': 'No data provided',
                        'message': 'Please provide JSON data with economic indicators'
                    }), 400
                
                # Validate input features
                missing_features = []
                for feature in self.feature_columns:
                    if feature not in data:
                        missing_features.append(feature)
                
                if missing_features:
                    return jsonify({
                        'error': 'Missing required features',
                        'missing_features': missing_features,
                        'required_features': self.feature_columns
                    }), 400
                
                # Prepare input data
                input_data = np.array([[data[feature] for feature in self.feature_columns]])
                
                # Scale the input data
                input_scaled = self.scaler.transform(input_data)
                
                # Make predictions
                revenue_prediction = float(self.revenue_model.predict(input_scaled)[0])
                profit_prediction = float(self.profit_model.predict(input_scaled)[0])
                
                # Calculate additional metrics
                profit_margin = (profit_prediction / revenue_prediction * 100) if revenue_prediction > 0 else 0
                
                response = {
                    'predictions': {
                        'revenue': round(revenue_prediction, 2),
                        'profit': round(profit_prediction, 2),
                        'profit_margin_percent': round(profit_margin, 2)
                    },
                    'input_data': data,
                    'model_accuracy': {
                        'revenue_accuracy': self.model_info.get('revenue_accuracy'),
                        'profit_accuracy': self.model_info.get('profit_accuracy'),
                        'overall_accuracy': self.model_info.get('overall_accuracy')
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Prediction made: Revenue=${revenue_prediction:.2f}, Profit=${profit_prediction:.2f}")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({
                    'error': 'Prediction failed',
                    'message': str(e)
                }), 500
        
        @self.app.route('/predict/sample', methods=['GET'])
        def predict_sample():
            """Make a prediction using sample data for testing"""
            if not self.models_loaded():
                return jsonify({
                    'error': 'Models not loaded'
                }), 500
            
            # Sample Sri Lankan economic data
            sample_data = {
                'gdp_growth': 3.2,
                'inflation_rate': 5.1,
                'interest_rate': 8.3,
                'exchange_rate': 315.5,
                'export_volume': 12500,
                'import_volume': 18200,
                'tourism_arrivals': 155000,
                'remittances': 675,
                'fdi_inflow': 890,
                'government_spending': 2600
            }
            
            # Make prediction using the predict route logic
            input_data = np.array([[sample_data[feature] for feature in self.feature_columns]])
            input_scaled = self.scaler.transform(input_data)
            
            revenue_prediction = float(self.revenue_model.predict(input_scaled)[0])
            profit_prediction = float(self.profit_model.predict(input_scaled)[0])
            profit_margin = (profit_prediction / revenue_prediction * 100) if revenue_prediction > 0 else 0
            
            return jsonify({
                'sample_prediction': {
                    'revenue': round(revenue_prediction, 2),
                    'profit': round(profit_prediction, 2),
                    'profit_margin_percent': round(profit_margin, 2)
                },
                'sample_input': sample_data,
                'note': 'This is a sample prediction using typical Sri Lankan economic indicators'
            })
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'available_endpoints': [
                    '/health',
                    '/model-details',
                    '/predict',
                    '/predict/sample'
                ]
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500
    
    def models_loaded(self):
        """Check if all models are loaded successfully"""
        return all([
            self.revenue_model is not None,
            self.profit_model is not None,
            self.scaler is not None,
            self.model_info is not None
        ])
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the Flask application"""
        print("🚀 LankaForecaster API Starting...")
        print("=" * 50)
        
        if self.models_loaded():
            print("✅ Revenue Model: Loaded with joblib")
            print("✅ Profit Model: Loaded with joblib") 
            print("✅ Scaler: Loaded with joblib")
            print(f"🎯 Model Accuracy: {self.model_info.get('overall_accuracy', 'N/A')}")
        else:
            print("❌ Models failed to load!")
        
        print("🌐 API Endpoints:")
        print(f"   - http://localhost:{port}/health")
        print(f"   - http://localhost:{port}/predict")
        print(f"   - http://localhost:{port}/model-details")
        print(f"   - http://localhost:{port}/predict/sample")
        print("=" * 50)
        
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function to run the API"""
    api = LankaForecasterAPI()
    api.run()

if __name__ == '__main__':
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 5000))
    api = LankaForecasterAPI()
    api.run(host='0.0.0.0', port=port, debug=False)