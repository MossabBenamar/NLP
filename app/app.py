from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json

def create_app(api_handler):
    """Create and configure the Flask application.
    
    Args:
        api_handler: An instance of the APIHandler class.
        
    Returns:
        The configured Flask application.
    """
    app = Flask(__name__)
    
    # Configure the application
    app.config['JSON_SORT_KEYS'] = False
    
    # Define routes
    @app.route('/')
    def index():
        """Render the main page of the application."""
        return render_template('index.html')
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze_code():
        """API endpoint for analyzing code and generating solutions."""
        try:
            # Get the request data
            request_data = request.get_json()
            
            # Process the request using the API handler
            return api_handler.process_request(request_data)
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/examples/<example_id>', methods=['GET'])
    def get_example(example_id):
        """API endpoint for retrieving example code and error messages."""
        return api_handler.get_example_code(example_id)
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory('static', filename)
    
    return app