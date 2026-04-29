#!/usr/bin/env python3
"""
Entry point to run the Flask dashboard
Usage: python run_dashboard.py
"""

from app.main import create_app

if __name__ == '__main__':
    app = create_app()
    print("🚀 Starting Flask dashboard at http://localhost:5000")
    print("Press CTRL+C to stop the server")
    app.run(debug=True, host='127.0.0.1', port=5000)
