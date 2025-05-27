import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class ContextAnalyzer:
    """A class for analyzing code context to understand the root cause of errors."""
    
    def __init__(self):
        """Initialize the context analyzer with error patterns and root cause templates."""
        # Define patterns for common error contexts
        self.context_patterns = {
            'syntax_error': {
                'missing_parenthesis': r'\(\s*[^\(\)]*$|^[^\(\)]*\s*\)',
                'missing_bracket': r'\[\s*[^\[\]]*$|^[^\[\]]*\s*\]',
                'missing_brace': r'\{\s*[^\{\}]*$|^[^\{\}]*\s*\}',
                'missing_colon': r'(if|else|elif|for|while|def|class)\s+[^:]*$',
                'invalid_indentation': r'^\s*\S+.*\n^(?!\s)',
            },
            'type_error': {
                'string_as_number': r'["\']\d+["\']\s*[\+\-\*\/]',
                'none_operation': r'None\s*[\+\-\*\/\[\]]',
                'wrong_function_args': r'\w+\([^\)]*\)\s*\.',
                'non_iterable': r'for\s+\w+\s+in\s+(\d+|True|False|None)',
            },
            'name_error': {
                'undefined_variable': r'\b(\w+)\b(?!\s*[=:\(\[\{])',
                'misspelled_variable': r'\b\w{3,}\b',
                'wrong_scope': r'def\s+\w+\([^\)]*\):\s*[^\n]*\n(?:\s+[^\n]*\n)*\s+return\s+\w+',
            },
            'index_error': {
                'out_of_bounds': r'\w+\s*\[\s*\d+\s*\]',
                'empty_list': r'\[\s*\]\s*\[',
                'wrong_loop_condition': r'for\s+\w+\s+in\s+range\(.*\):\s*[^\n]*\n(?:\s+[^\n]*\n)*\s+\w+\[\w+\]',
            },
            'key_error': {
                'missing_key': r'\w+\s*\[\s*["\']\w+["\']\s*\]',
                'wrong_key_type': r'\w+\s*\[\s*\w+\s*\]',
            },
            'division_by_zero': {
                'explicit_zero_division': r'\s*\/\s*0',
                'variable_zero_division': r'\s*\/\s*\w+',
            },
            'attribute_error': {
                'undefined_attribute': r'\w+\s*\.\s*\w+',
                'none_attribute': r'None\s*\.\s*\w+',
            },
        }
        
        # Root cause templates for different error types
        self.root_cause_templates = {
            'syntax_error': [
                "Missing or unmatched parenthesis, bracket, or brace",
                "Missing colon after control statement",
                "Invalid indentation",
                "Invalid syntax in the code structure"
            ],
            'type_error': [
                "Operation between incompatible types",
                "Attempting to use None in an operation",
                "Passing wrong type of argument to a function",
                "Trying to iterate over a non-iterable object"
            ],
            'name_error': [
                "Using a variable that is not defined",
                "Misspelling a variable name",
                "Using a variable outside its scope",
                "Forgetting to import a required module"
            ],
            'index_error': [
                "Accessing an index that is out of range",
                "Trying to access an element from an empty list",
                "Off-by-one error in a loop",
                "Using an incorrect loop termination condition"
            ],
            'key_error': [
                "Trying to access a dictionary key that doesn't exist",
                "Misspelling a dictionary key",
                "Using a key of the wrong type",
                "Assuming a key exists without checking first"
            ],
            'division_by_zero': [
                "Dividing by zero explicitly",
                "Dividing by a variable that has a value of zero",
                "Not checking for zero before division",
                "Logic error leading to a zero denominator"
            ],
            'attribute_error': [
                "Accessing an attribute that doesn't exist",
                "Trying to access an attribute on None",
                "Misspelling an attribute name",
                "Using an attribute before it's defined"
            ],
        }
        
        # Explanation templates for different error types
        self.explanation_templates = {
            'syntax_error': [
                "Your code has a syntax error. This means the structure of your code doesn't follow the rules of the programming language. Check for missing or mismatched parentheses, brackets, braces, or colons.",
                "Syntax errors occur when the code doesn't conform to the language's grammar rules. Look for incorrect indentation, missing punctuation, or invalid statements."
            ],
            'type_error': [
                "A type error occurs when you try to perform an operation on a value of the wrong type. For example, trying to add a string and a number without conversion, or calling a method on an object that doesn't support it.",
                "Your code is trying to use a value in a way that's not compatible with its type. Check that variables have the expected types before operations."
            ],
            'name_error': [
                "A name error happens when you try to use a variable or function that hasn't been defined yet. Make sure all variables are defined before use and check for typos in variable names.",
                "Your code references a name that Python doesn't recognize. This could be because the variable isn't defined, is misspelled, or is used outside its scope."
            ],
            'index_error': [
                "An index error occurs when you try to access an element at an index that doesn't exist in a list or array. Remember that indices start at 0 and the valid range is 0 to length-1.",
                "Your code is trying to access an element at a position that's outside the bounds of the list or array. Check your loop conditions and make sure you're not trying to access elements beyond the end of the collection."
            ],
            'key_error': [
                "A key error happens when you try to access a dictionary using a key that doesn't exist. Make sure the key exists before trying to access it, or use methods like .get() that handle missing keys gracefully.",
                "Your code is trying to access a dictionary with a key that isn't present. Consider using the 'in' operator to check if a key exists before accessing it."
            ],
            'division_by_zero': [
                "Division by zero is a mathematical error that occurs when you try to divide a number by zero. Always check that your denominator is not zero before performing division.",
                "Your code is attempting to divide by zero, which is mathematically undefined. Add a condition to check if the divisor is zero before performing the division operation."
            ],
            'attribute_error': [
                "An attribute error occurs when you try to access an attribute or method that doesn't exist on an object. Check that the object is of the expected type and that the attribute name is spelled correctly.",
                "Your code is trying to access a property or method that doesn't exist on the object. This could be because the object is None, is of the wrong type, or the attribute name is misspelled."
            ],
        }
    
    def analyze(self, preprocessed_data, error_type):
        """Analyze the code context to understand the root cause of the error.
        
        Args:
            preprocessed_data: A dictionary containing preprocessed error message and code context.
            error_type: The classified error type.
            
        Returns:
            A dictionary containing the analysis results, including root cause and explanation.
        """
        error_message = preprocessed_data.get('error_message', '')
        code_context = preprocessed_data.get('code_context', '')
        line_number = preprocessed_data.get('line_number', None)
        
        # Get the patterns for the specific error type
        patterns = self.context_patterns.get(error_type, {})
        
        # Find matches for each pattern in the code context
        matches = {}
        for pattern_name, pattern in patterns.items():
            matches[pattern_name] = re.findall(pattern, code_context, re.MULTILINE)
        
        # Determine the most likely root cause based on the matches
        root_cause = self._determine_root_cause(error_type, matches, error_message)
        
        # Generate an explanation for the error
        explanation = self._generate_explanation(error_type, root_cause, matches)
        
        return {
            'error_type': error_type,
            'root_cause': root_cause,
            'explanation': explanation,
            'matches': matches,
            'line_number': line_number
        }
    
    def _determine_root_cause(self, error_type, matches, error_message):
        """Determine the most likely root cause based on pattern matches.
        
        Args:
            error_type: The classified error type.
            matches: Dictionary of pattern matches.
            error_message: The error message string.
            
        Returns:
            A string describing the root cause of the error.
        """
        # If we have matches for any patterns, use them to determine the root cause
        for pattern_name, match_list in matches.items():
            if match_list:
                # Use the pattern name to create a specific root cause message
                if pattern_name == 'missing_parenthesis':
                    return "Missing or unmatched parenthesis in the code"
                elif pattern_name == 'missing_bracket':
                    return "Missing or unmatched bracket in the code"
                elif pattern_name == 'missing_brace':
                    return "Missing or unmatched brace in the code"
                elif pattern_name == 'missing_colon':
                    return "Missing colon after a control statement"
                elif pattern_name == 'invalid_indentation':
                    return "Invalid indentation in the code"
                elif pattern_name == 'string_as_number':
                    return "Attempting to use a string as a number without conversion"
                elif pattern_name == 'none_operation':
                    return "Performing an operation on None"
                elif pattern_name == 'wrong_function_args':
                    return "Passing incorrect arguments to a function"
                elif pattern_name == 'non_iterable':
                    return "Trying to iterate over a non-iterable object"
                elif pattern_name == 'undefined_variable':
                    return "Using a variable that is not defined"
                elif pattern_name == 'misspelled_variable':
                    return "Possible misspelling of a variable name"
                elif pattern_name == 'wrong_scope':
                    return "Using a variable outside its scope"
                elif pattern_name == 'out_of_bounds':
                    return "Accessing an index that is out of range"
                elif pattern_name == 'empty_list':
                    return "Trying to access an element from an empty list"
                elif pattern_name == 'wrong_loop_condition':
                    return "Incorrect loop termination condition"
                elif pattern_name == 'missing_key':
                    return "Trying to access a dictionary key that doesn't exist"
                elif pattern_name == 'wrong_key_type':
                    return "Using a key of the wrong type"
                elif pattern_name == 'explicit_zero_division':
                    return "Dividing by zero explicitly"
                elif pattern_name == 'variable_zero_division':
                    return "Dividing by a variable that has a value of zero"
                elif pattern_name == 'undefined_attribute':
                    return "Accessing an attribute that doesn't exist"
                elif pattern_name == 'none_attribute':
                    return "Trying to access an attribute on None"
        
        # If no specific pattern matches, use a generic template for the error type
        templates = self.root_cause_templates.get(error_type, [])
        if templates:
            # For simplicity, just return the first template
            # In a real system, you might use more sophisticated selection
            return templates[0]
        
        # Default message if no templates are available
        return f"An error of type '{error_type}' occurred in the code"
    
    def _generate_explanation(self, error_type, root_cause, matches):
        """Generate an explanation for the error based on the root cause.
        
        Args:
            error_type: The classified error type.
            root_cause: The determined root cause.
            matches: Dictionary of pattern matches.
            
        Returns:
            A string explaining the error and its cause.
        """
        # Get explanation templates for the error type
        templates = self.explanation_templates.get(error_type, [])
        
        if templates:
            # For simplicity, just use the first template
            # In a real system, you might select based on the specific root cause
            explanation = templates[0]
        else:
            # Default explanation if no templates are available
            explanation = f"An error of type '{error_type}' occurred in your code. "
            explanation += f"The root cause appears to be: {root_cause}."
        
        # Add specific details based on the matches if available
        for pattern_name, match_list in matches.items():
            if match_list and len(match_list) > 0:
                if pattern_name == 'undefined_variable' and len(match_list[0]) > 0:
                    explanation += f" The variable '{match_list[0]}' might be undefined or misspelled."
                elif pattern_name == 'out_of_bounds' and len(match_list) > 0:
                    explanation += f" The index in '{match_list[0]}' might be out of range."
                elif pattern_name == 'missing_key' and len(match_list) > 0:
                    explanation += f" The key in '{match_list[0]}' might not exist in the dictionary."
        
        return explanation