"""
Flask application for Fintech-LLM Stock Forecast Dashboard
This is the main entry point where the Flask app is created and configured
"""

from flask import Flask
from app.routes.forecast import forecast_bp


def create_app():
    """
    Application factory function
    This creates and configures the Flask app
    Benefits: easier testing, multiple configs, modular structure
    """
    app = Flask(__name__)
    
    # Configuration
    app.config['JSON_SORT_KEYS'] = False
    
    # Register blueprints (organized route groups)
    # A blueprint is like a module for routes
    app.register_blueprint(forecast_bp)
    
    return app


if __name__ == '__main__':
    app = create_app()
    # debug=True: auto-reloads when you change code
    app.run(debug=True, host='0.0.0.0', port=5000)
