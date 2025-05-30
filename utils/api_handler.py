import json
from flask import jsonify

class APIHandler:
    """A class for handling API requests and responses."""
    
    def __init__(self, preprocessor, error_classifier, context_analyzer, solution_generator, ml_model=None):
        """Initialize the API handler with the necessary components.
        
        Args:
            preprocessor: An instance of the Preprocessor class.
            error_classifier: An instance of the ErrorClassifier class.
            context_analyzer: An instance of the ContextAnalyzer class.
            solution_generator: An instance of the SolutionGenerator class.
            ml_model: An instance of the CodeBERTModel class (which uses Google Gemini-2.0-flash internally) (optional).
        """
        self.preprocessor = preprocessor
        self.error_classifier = error_classifier
        self.context_analyzer = context_analyzer
        self.solution_generator = solution_generator
        self.ml_model = ml_model
    
    def process_request(self, request_data):
        """Process an API request and generate a response.
        
        Args:
            request_data: A dictionary containing the request data.
            
        Returns:
            A Flask response object containing the analysis results.
        """
        try:
            # Extract data from the request
            code = request_data.get('code', '')
            error_message = request_data.get('error_message', '')
            language = request_data.get('language', 'python')
            
            # Validate the request data
            if not code:
                return jsonify({
                    'success': False,
                    'error': 'Code is required'
                }), 400
            
            # Preprocess the code and error message
            preprocessed_data = self.preprocessor.preprocess(code, error_message, language)
            
            # Classify the error type
            # Try ML-based classification first if available
            error_type = None
            confidence = 0.0
            suggested_fix = None
            if self.ml_model and hasattr(self.ml_model, 'classify'):
                try:
                    # The classify method now returns a tuple (error_type, confidence, suggested_fix)
                    classification_result = self.ml_model.classify(
                        preprocessed_data.get('normalized_code', ''),
                        preprocessed_data.get('error_message', '')
                    )
                    
                    # Handle both old and new return formats
                    if isinstance(classification_result, tuple):
                        if len(classification_result) >= 1:
                            error_type = classification_result[0]
                        if len(classification_result) >= 2:
                            confidence = classification_result[1]
                        if len(classification_result) >= 3:
                            suggested_fix = classification_result[2]
                    else:
                        # Handle old format (just a string)
                        error_type = classification_result
                except Exception as e:
                    print(f"ML classification failed: {e}")
            
            # Fall back to rule-based classification if ML fails or is not available
            if not error_type:
                error_type = self.error_classifier.classify(preprocessed_data)
            
            # Analyze the code context to determine the root cause
            context_analysis = self.context_analyzer.analyze(preprocessed_data, error_type)
            
            # Generate solution recommendations
            solutions = []
            
            # Use the suggested_fix from classification if available
            if suggested_fix:
                solutions.append({
                    'description': 'Rule-based fix suggestion',
                    'code': suggested_fix,
                    'confidence': f'{confidence:.2f}',
                    'source': 'ml_model_rule_based'
                })
            
            # Try ML-based solution generation if rule-based fix isn't available
            elif self.ml_model and hasattr(self.ml_model, 'generate_fix'):
                try:
                    ml_solution = self.ml_model.generate_fix(
                        preprocessed_data.get('normalized_code', ''),
                        error_type,
                        preprocessed_data.get('error_message', ''),
                        preprocessed_data.get('line_number')
                    )
                    if ml_solution:
                        solutions.append({
                            'description': 'AI-generated solution',
                            'code': ml_solution,
                            'confidence': 'high',
                            'source': 'ml_model'
                        })
                except Exception as e:
                    print(f"ML solution generation failed: {e}")
            
            # Always get rule-based solutions as well
            rule_based_solutions = self.solution_generator.generate(preprocessed_data, error_type, context_analysis)
            solutions.extend(rule_based_solutions)
            
            # Prepare the response
            response = {
                'success': True,
                'analysis': {
                    'error_type': error_type,
                    'confidence': confidence if confidence > 0 else None,
                    'root_cause': context_analysis.get('root_cause', 'Unknown'),
                    'line_number': preprocessed_data.get('line_number'),
                    'code_context': preprocessed_data.get('code_context')
                },
                'solutions': solutions
            }
            
            return jsonify(response)
        
        except Exception as e:
            # Handle any exceptions that occur during processing
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def get_example_code(self, example_id):
        """Get example code and error message for demonstration purposes.
        
        Args:
            example_id: The ID of the example to retrieve.
            
        Returns:
            A Flask response object containing the example code and error message.
        """
        examples = {
            'syntax_error': {
                'code': 'def calculate_sum(a, b)\n    return a + b\n\nresult = calculate_sum(5, 10)\nprint(result)',
                'error_message': 'SyntaxError: expected \':\' at line 1',
                'language': 'python',
                'description': 'Missing colon after function definition'
            },
            'type_error': {
                'code': 'def calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)\n\ndata = [10, 20, 30, "40", 50]\nresult = calculate_average(data)\nprint(result)',
                'error_message': 'TypeError: unsupported operand type(s) for +: \'int\' and \'str\' at line 2',
                'language': 'python',
                'description': 'Trying to sum a list containing a string'
            },
            'name_error': {
                'code': 'def calculate_area(radius):\n    area = pi * radius * radius\n    return area\n\nresult = calculate_area(5)\nprint(result)',
                'error_message': 'NameError: name \'pi\' is not defined at line 2',
                'language': 'python',
                'description': 'Using an undefined variable (pi)'
            },
            'index_error': {
                'code': 'def get_element(list_data, index):\n    return list_data[index]\n\nmy_list = [10, 20, 30]\nresult = get_element(my_list, 5)\nprint(result)',
                'error_message': 'IndexError: list index out of range at line 2',
                'language': 'python',
                'description': 'Accessing a list with an index that is out of bounds'
            },
            'key_error': {
                'code': 'def get_value(dict_data, key):\n    return dict_data[key]\n\nmy_dict = {"a": 10, "b": 20, "c": 30}\nresult = get_value(my_dict, "d")\nprint(result)',
                'error_message': 'KeyError: \'d\' at line 2',
                'language': 'python',
                'description': 'Accessing a dictionary with a key that does not exist'
            },
            'division_by_zero': {
                'code': 'def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)\nprint(result)',
                'error_message': 'ZeroDivisionError: division by zero at line 2',
                'language': 'python',
                'description': 'Dividing a number by zero'
            },
            'attribute_error': {
                'code': 'class Person:\n    def __init__(self, name):\n        self.name = name\n\nperson = Person("John")\nprint(person.age)',
                'error_message': 'AttributeError: \'Person\' object has no attribute \'age\' at line 6',
                'language': 'python',
                'description': 'Accessing an attribute that does not exist on an object'
            },
            'javascript_syntax': {
                'code': 'function calculateSum(a, b) {\n  return a + b\n}\n\nconst result = calculateSum(5, 10);\nconsole.log(result);',
                'error_message': 'SyntaxError: missing semicolon at line 2',
                'language': 'javascript',
                'description': 'Missing semicolon in JavaScript'
            },
            'javascript_reference': {
                'code': 'function displayMessage() {\n  console.log(message);\n}\n\ndisplayMessage();',
                'error_message': 'ReferenceError: message is not defined at line 2',
                'language': 'javascript',
                'description': 'Using an undefined variable in JavaScript'
            }
        }
        
        if example_id == 'all':
            # Return all examples as a list
            examples_list = []
            for key, example in examples.items():
                example_with_id = example.copy()
                example_with_id['id'] = key
                examples_list.append(example_with_id)
            return jsonify(examples_list)
        elif example_id in examples:
            return jsonify({
                'success': True,
                'example': examples[example_id]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Example not found'
            }), 404