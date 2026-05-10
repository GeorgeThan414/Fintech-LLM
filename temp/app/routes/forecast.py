"""
Routes for forecast dashboard
A route is a URL pattern that maps to a function
Example: /dashboard -> function that returns dashboard HTML
"""

from flask import Blueprint, render_template
import csv
import os

# Create a blueprint
# Blueprints let you organize routes into modules
forecast_bp = Blueprint('forecast', __name__)


def load_forecast_data():
    """
    Load forecast results from CSV
    Returns a list of forecast dictionaries
    """
    csv_path = 'forecast_results_FinGPT.csv'
    
    if not os.path.exists(csv_path):
        return []
    
    forecasts = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse headlines into list (they're pipe-separated)
                headlines = row['headlines'].split(' | ') if row['headlines'] else []
                
                forecasts.append({
                    'ticker': row['ticker'],
                    'company': row['company'],
                    'prediction': row['prediction'].lower(),  # rise, fall, remain
                    'last_close': float(row['last_close']),
                    'headlines': headlines[:5],  # Top 5 headlines
                    'raw_output': row['raw_output'],
                })
    except Exception as e:
        print(f"Error reading CSV: {e}")
    
    return forecasts


@forecast_bp.route('/')
def index():
    """
    Home page - redirects to dashboard
    """
    return '''
    <h1>Welcome to Stock Forecast Dashboard</h1>
    <p><a href="/dashboard">Go to Dashboard</a></p>
    '''


@forecast_bp.route('/dashboard')
def dashboard():
    """
    Main dashboard view
    Shows all forecasts in a table/card format
    
    render_template: loads HTML file and passes data to it
    """
    forecasts = load_forecast_data()
    
    # Get prediction stats
    stats = {
        'total': len(forecasts),
        'rise': len([f for f in forecasts if f['prediction'] == 'rise']),
        'fall': len([f for f in forecasts if f['prediction'] == 'fall']),
        'remain': len([f for f in forecasts if f['prediction'] == 'remain']),
    }
    
    return render_template('dashboard.html', forecasts=forecasts, stats=stats)


@forecast_bp.route('/forecast/<ticker>')
def forecast_detail(ticker):
    """
    Individual stock detail page
    Shows full analysis for a specific stock
    
    <ticker> is a URL parameter - captured from the URL
    Example: /forecast/TSLA -> ticker = "TSLA"
    """
    forecasts = load_forecast_data()
    
    # Find the forecast for this ticker
    forecast = next((f for f in forecasts if f['ticker'] == ticker), None)
    
    if not forecast:
        return f"Forecast not found for {ticker}", 404
    
    return render_template('detail.html', forecast=forecast)
