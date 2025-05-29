import os
from app.app import create_app
from utils.preprocessor import Preprocessor
from models.error_classifier import ErrorClassifier
from models.context_analyzer import ContextAnalyzer
from models.solution_gen import SolutionGenerator
from models.ml_model import CodeBERTModel  # Now using DialoGPT model internally
from utils.api_handler import APIHandler

def main():
    """Initialize and run the NLP Code Debugger application."""
    # Initialize the components
    preprocessor = Preprocessor()
    error_classifier = ErrorClassifier()
    context_analyzer = ContextAnalyzer()
    solution_generator = SolutionGenerator()
    ml_model = CodeBERTModel()  # Initialize the ML model
    
    # Initialize the API handler
    api_handler = APIHandler(
        preprocessor=preprocessor,
        error_classifier=error_classifier,
        context_analyzer=context_analyzer,
        solution_generator=solution_generator,
        ml_model=ml_model
    )
    
    # Create the Flask application
    app = create_app(api_handler)
    
    # Get the port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the application
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    main()