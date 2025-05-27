import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download NLTK resources (uncomment first time)
# nltk.download('punkt')
# nltk.download('stopwords')

class ErrorClassifier:
    """A class for classifying programming errors based on error messages and code context."""
    
    def __init__(self):
        """Initialize the error classifier with predefined error types and patterns."""
        # Define common error types and their patterns
        self.error_types = {
            'syntax_error': [
                r'syntax\s+error', r'invalid\s+syntax', r'unexpected', 
                r'token\s+error', r'parsing\s+error', r'SyntaxError', r'expected'
            ],
            'type_error': [
                r'type\s+error', r'cannot\s+convert', r'not\s+iterable',
                r'not\s+callable', r'not\s+subscriptable', r'NoneType', r'TypeError',
                r'unsupported\s+operand', r'object\s+is\s+not', r'must\s+be'
            ],
            'name_error': [
                r'name\s+error', r'undefined', r'not\s+defined', 
                r'unknown\s+variable', r'unknown\s+identifier', r'NameError'
            ],
            'index_error': [
                r'index\s+error', r'out\s+of\s+range', r'index\s+out\s+of\s+bounds',
                r'array\s+index\s+out', r'list\s+index\s+out', r'IndexError'
            ],
            'key_error': [
                r'key\s+error', r'no\s+such\s+key', r'key\s+not\s+found',
                r'invalid\s+key', r'missing\s+key', r'KeyError', r'dictionary'
            ],
            'value_error': [
                r'value\s+error', r'invalid\s+value', r'invalid\s+literal',
                r'invalid\s+argument', r'invalid\s+parameter'
            ],
            'attribute_error': [
                r'attribute\s+error', r'no\s+attribute', r'has\s+no\s+attribute',
                r'undefined\s+property', r'unknown\s+property'
            ],
            'import_error': [
                r'import\s+error', r'no\s+module', r'cannot\s+find\s+module',
                r'module\s+not\s+found', r'package\s+not\s+found'
            ],
            'io_error': [
                r'io\s+error', r'file\s+not\s+found', r'no\s+such\s+file',
                r'permission\s+denied', r'access\s+denied'
            ],
            'memory_error': [
                r'memory\s+error', r'out\s+of\s+memory', r'memory\s+allocation',
                r'stack\s+overflow', r'heap\s+overflow'
            ],
            'runtime_error': [
                r'runtime\s+error', r'exception\s+occurred', r'unexpected\s+error',
                r'fatal\s+error', r'critical\s+error'
            ],
            'logic_error': [
                r'logic\s+error', r'assertion\s+error', r'assertion\s+failed',
                r'condition\s+failed', r'invariant\s+violated'
            ],
            'null_pointer': [
                r'null\s+pointer', r'null\s+reference', r'NullPointerException',
                r'dereferencing\s+null', r'null\s+object'
            ],
            'division_by_zero': [
                r'division\s+by\s+zero', r'divide\s+by\s+zero', r'zero\s+division',
                r'divided\s+by\s+zero', r'modulo\s+by\s+zero'
            ],
            'overflow_error': [
                r'overflow', r'integer\s+overflow', r'arithmetic\s+overflow',
                r'buffer\s+overflow', r'stack\s+overflow'
            ],
            'timeout_error': [
                r'timeout', r'time\s+limit', r'execution\s+time',
                r'deadline\s+exceeded', r'request\s+timeout'
            ],
            'connection_error': [
                r'connection\s+error', r'network\s+error', r'socket\s+error',
                r'connection\s+refused', r'connection\s+timeout'
            ],
            'permission_error': [
                r'permission\s+error', r'access\s+denied', r'forbidden',
                r'unauthorized', r'not\s+allowed'
            ],
            'dependency_error': [
                r'dependency\s+error', r'version\s+conflict', r'incompatible',
                r'missing\s+dependency', r'library\s+conflict'
            ]
        }
        
        # Initialize the ML model (placeholder for future implementation)
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize and train the machine learning model.
        
        In a real implementation, this would load a pre-trained model or train one
        on a dataset of error messages and their classifications.
        """
        # This is a placeholder for a real ML model
        # In a production system, you would load a pre-trained model here
        # or train one on a dataset of error messages and their classifications
        
        # Example of a simple pipeline that could be used
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        # In a real implementation, you would train the model here
        # self.model.fit(X_train, y_train)
    
    def classify(self, preprocessed_data):
        """Classify the error type based on the preprocessed data.
        
        Args:
            preprocessed_data: A dictionary containing preprocessed code and error information.
            
        Returns:
            A string representing the classified error type, or 'unknown' if the error cannot be classified.
        """
        error_message = preprocessed_data.get('error_message', '')
        code_context = preprocessed_data.get('code_context', '')
        
        # Direct error type detection based on error message
        if 'SyntaxError' in error_message:
            return 'syntax_error'
        elif 'TypeError' in error_message:
            return 'type_error'
        elif 'KeyError' in error_message:
            return 'key_error'
        elif 'IndexError' in error_message:
            return 'index_error'
        elif 'NameError' in error_message:
            return 'name_error'
        elif 'ZeroDivisionError' in error_message:
            return 'division_by_zero'
        elif 'AttributeError' in error_message:
            return 'attribute_error'
        elif 'ReferenceError' in error_message:
            return 'reference_error'
        
        # If no error message is provided, return unknown
        if not error_message:
            return 'unknown'
        
        # If we have a trained model, use it for classification
        if self.model and hasattr(self.model, 'predict') and False:  # Disabled for now
            # Combine error message and code context for prediction
            combined_text = f"{error_message} {code_context}"
            # Make prediction
            prediction = self.model.predict([combined_text])[0]
            return prediction
        
        # Fallback to rule-based classification
        return self._classify_with_rules(error_message, code_context)
    
    def _classify_with_rules(self, error_message, code_context):
        """Classify the error type based on pattern matching in the error message and code context.
        
        Args:
            error_message: The error message string.
            code_context: The code context around the error.
            
        Returns:
            A string representing the classified error type.
        """
        # First check for exact error type names in the original case
        if 'SyntaxError' in error_message:
            return 'syntax_error'
        elif 'TypeError' in error_message:
            return 'type_error'
        elif 'KeyError' in error_message:
            return 'key_error'
        elif 'IndexError' in error_message:
            return 'index_error'
        elif 'NameError' in error_message:
            return 'name_error'
        elif 'ZeroDivisionError' in error_message:
            return 'division_by_zero'
        elif 'AttributeError' in error_message:
            return 'attribute_error'
        elif 'ReferenceError' in error_message:
            return 'reference_error'
        
        # Check for common error keywords in the error message
        error_message_lower = error_message.lower()
        if 'syntax' in error_message_lower or 'invalid syntax' in error_message_lower or 'unexpected' in error_message_lower:
            return 'syntax_error'
        elif 'type' in error_message_lower or 'cannot convert' in error_message_lower or 'not iterable' in error_message_lower:
            return 'type_error'
        elif 'key' in error_message_lower or 'dictionary' in error_message_lower:
            return 'key_error'
            
        # Combine error message and code context for pattern matching
        combined_text = f"{error_message} {code_context}".lower()
        
        # Check each error type's patterns
        for error_type, patterns in self.error_types.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    return error_type
        
        # Default to 'unknown_error' if no patterns match
        return 'unknown_error'
    
    def get_error_details(self, error_type):
        """Get detailed information about an error type.
        
        Args:
            error_type: The classified error type.
            
        Returns:
            A dictionary containing details about the error type.
        """
        error_details = {
            'syntax_error': {
                'description': 'Error in the syntax or structure of the code',
                'common_causes': [
                    'Missing parentheses, brackets, or braces',
                    'Incorrect indentation',
                    'Missing colons in Python',
                    'Invalid operators or expressions'
                ]
            },
            'type_error': {
                'description': 'Operation applied to an object of inappropriate type',
                'common_causes': [
                    'Trying to perform operations on incompatible types',
                    'Passing wrong type of argument to a function',
                    'Using a non-callable object as a function',
                    'Accessing a non-subscriptable object with an index'
                ]
            },
            'name_error': {
                'description': 'Attempt to access a variable or function that does not exist',
                'common_causes': [
                    'Using a variable before it is defined',
                    'Misspelling a variable or function name',
                    'Using a variable outside its scope',
                    'Forgetting to import a module'
                ]
            },
            'index_error': {
                'description': 'Attempt to access an index that is outside the bounds of a list or array',
                'common_causes': [
                    'Using an index that is negative or too large',
                    'Off-by-one errors in loops',
                    'Empty lists or arrays',
                    'Incorrect loop termination conditions'
                ]
            },
            'key_error': {
                'description': 'Attempt to access a dictionary with a key that does not exist',
                'common_causes': [
                    'Using a key that does not exist in the dictionary',
                    'Misspelling a key name',
                    'Case sensitivity issues with keys',
                    'Assuming a key exists without checking'
                ]
            },
            # Add more error types as needed
        }
        
        # Return details for the specified error type, or a default message
        return error_details.get(error_type, {
            'description': 'An error occurred in the code',
            'common_causes': ['Various issues in the code logic or syntax']
        })