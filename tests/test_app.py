import unittest
import json
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.app import create_app
from utils.preprocessor import Preprocessor
from models.error_classifier import ErrorClassifier
from models.context_analyzer import ContextAnalyzer
from models.solution_gen import SolutionGenerator
from utils.api_handler import APIHandler

class TestApp(unittest.TestCase):
    """Test cases for the NLP Code Debugger application."""
    
    def setUp(self):
        """Set up the test environment."""
        # Initialize the components
        self.preprocessor = Preprocessor()
        self.error_classifier = ErrorClassifier()
        self.context_analyzer = ContextAnalyzer()
        self.solution_generator = SolutionGenerator()
        
        # Initialize the API handler
        self.api_handler = APIHandler(
            preprocessor=self.preprocessor,
            error_classifier=self.error_classifier,
            context_analyzer=self.context_analyzer,
            solution_generator=self.solution_generator
        )
        
        # Create the Flask application
        self.app = create_app(self.api_handler)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_index_route(self):
        """Test the index route."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_analyze_code_syntax_error(self):
        """Test the analyze code endpoint with a syntax error."""
        data = {
            'code': 'def calculate_sum(a, b)\n    return a + b\n\nresult = calculate_sum(5, 10)\nprint(result)',
            'error_message': 'SyntaxError: expected \':\' at line 1',
            'language': 'python'
        }
        
        response = self.client.post(
            '/api/analyze',
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertEqual(response_data['analysis']['error_type'], 'syntax_error')
    
    def test_analyze_code_type_error(self):
        """Test the analyze code endpoint with a type error."""
        data = {
            'code': 'def calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)\n\ndata = [10, 20, 30, "40", 50]\nresult = calculate_average(data)\nprint(result)',
            'error_message': 'TypeError: unsupported operand type(s) for +: \'int\' and \'str\' at line 2',
            'language': 'python'
        }
        
        response = self.client.post(
            '/api/analyze',
            data=json.dumps(data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertEqual(response_data['analysis']['error_type'], 'type_error')
    
    def test_get_example(self):
        """Test the get example endpoint."""
        response = self.client.get('/api/examples/syntax_error')
        
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertEqual(response_data['example']['language'], 'python')
    
    def test_get_nonexistent_example(self):
        """Test the get example endpoint with a nonexistent example ID."""
        response = self.client.get('/api/examples/nonexistent_example')
        
        self.assertEqual(response.status_code, 404)
        response_data = json.loads(response.data)
        self.assertFalse(response_data['success'])

if __name__ == '__main__':
    unittest.main()