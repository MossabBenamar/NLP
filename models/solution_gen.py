import re

class SolutionGenerator:
    """A class for generating solutions to programming errors based on analysis."""
    
    def __init__(self):
        """Initialize the solution generator with solution templates for different error types."""
        # Solution templates for different error types
        self.solution_templates = {
            'syntax_error': {
                'missing_parenthesis': [
                    {
                        'description': 'Add the missing parenthesis',
                        'code_template': 'Replace {code_snippet} with {fixed_code}'
                    }
                ],
                'missing_bracket': [
                    {
                        'description': 'Add the missing bracket',
                        'code_template': 'Replace {code_snippet} with {fixed_code}'
                    }
                ],
                'missing_brace': [
                    {
                        'description': 'Add the missing brace',
                        'code_template': 'Replace {code_snippet} with {fixed_code}'
                    }
                ],
                'missing_colon': [
                    {
                        'description': 'Add a colon after the control statement',
                        'code_template': 'Replace {code_snippet} with {code_snippet}:'
                    }
                ],
                'invalid_indentation': [
                    {
                        'description': 'Fix the indentation',
                        'code_template': 'Ensure consistent indentation throughout your code'
                    }
                ],
                'default': [
                    {
                        'description': 'Check for missing punctuation or incorrect syntax',
                        'code_template': 'Review your code for syntax errors'
                    }
                ]
            },
            'type_error': {
                'string_as_number': [
                    {
                        'description': 'Convert the string to a number before performing arithmetic',
                        'code_template': 'Replace {code_snippet} with int({code_snippet}) or float({code_snippet})'
                    }
                ],
                'none_operation': [
                    {
                        'description': 'Check if the variable is None before performing operations',
                        'code_template': 'if {variable} is not None:\n    # Perform operation with {variable}'
                    }
                ],
                'wrong_function_args': [
                    {
                        'description': 'Check the function documentation for the correct arguments',
                        'code_template': 'Ensure the arguments passed to {function_name} are of the correct type'
                    }
                ],
                'non_iterable': [
                    {
                        'description': 'Ensure the object is iterable before using it in a loop',
                        'code_template': 'Make sure {variable} is a list, tuple, or other iterable type'
                    }
                ],
                'default': [
                    {
                        'description': 'Check the types of your variables before operations',
                        'code_template': 'Use type() to check variable types and convert if necessary'
                    }
                ]
            },
            'name_error': {
                'undefined_variable': [
                    {
                        'description': 'Define the variable before using it',
                        'code_template': '{variable_name} = value  # Define the variable first'
                    }
                ],
                'misspelled_variable': [
                    {
                        'description': 'Check for typos in variable names',
                        'code_template': '# Correct the spelling of the variable name'
                    }
                ],
                'wrong_scope': [
                    {
                        'description': 'Make sure the variable is accessible in the current scope',
                        'code_template': '# Define the variable in the appropriate scope or pass it as a parameter'
                    }
                ],
                'default': [
                    {
                        'description': 'Define all variables before using them',
                        'code_template': '# Ensure all variables are defined before use'
                    }
                ]
            },
            'index_error': {
                'out_of_bounds': [
                    {
                        'description': 'Check that the index is within the valid range',
                        'code_template': 'if 0 <= {index} < len({list_name}):\n    # Access {list_name}[{index}]'
                    }
                ],
                'empty_list': [
                    {
                        'description': 'Check if the list is empty before accessing elements',
                        'code_template': 'if {list_name}:\n    # Access elements of {list_name}'
                    }
                ],
                'wrong_loop_condition': [
                    {
                        'description': 'Fix the loop condition to prevent out-of-bounds access',
                        'code_template': 'for i in range(len({list_name})):\n    # Access {list_name}[i]'
                    }
                ],
                'default': [
                    {
                        'description': 'Ensure indices are within the valid range',
                        'code_template': '# Check list length before accessing elements'
                    }
                ]
            },
            'key_error': {
                'missing_key': [
                    {
                        'description': 'Check if the key exists before accessing it',
                        'code_template': 'if "{key}" in {dict_name}:\n    # Access {dict_name}["{key}"]'
                    },
                    {
                        'description': 'Use the .get() method to provide a default value',
                        'code_template': 'value = {dict_name}.get("{key}", default_value)'
                    }
                ],
                'wrong_key_type': [
                    {
                        'description': 'Ensure the key is of the correct type',
                        'code_template': '# Convert the key to the appropriate type'
                    }
                ],
                'default': [
                    {
                        'description': 'Check if keys exist before accessing them',
                        'code_template': '# Use the "in" operator or .get() method for safe dictionary access'
                    }
                ]
            },
            'division_by_zero': {
                'explicit_zero_division': [
                    {
                        'description': 'Avoid dividing by zero',
                        'code_template': '# Replace the zero divisor with a non-zero value'
                    }
                ],
                'variable_zero_division': [
                    {
                        'description': 'Check if the divisor is zero before dividing',
                        'code_template': 'if {divisor} != 0:\n    result = {dividend} / {divisor}\nelse:\n    # Handle the zero divisor case'
                    }
                ],
                'default': [
                    {
                        'description': 'Always check for zero before division',
                        'code_template': '# Add a condition to check for zero divisor'
                    }
                ]
            },
            'attribute_error': {
                'undefined_attribute': [
                    {
                        'description': 'Check if the attribute exists on the object',
                        'code_template': 'if hasattr({object}, "{attribute}"):\n    # Access {object}.{attribute}'
                    }
                ],
                'none_attribute': [
                    {
                        'description': 'Check if the object is None before accessing attributes',
                        'code_template': 'if {object} is not None:\n    # Access {object}.{attribute}'
                    }
                ],
                'default': [
                    {
                        'description': 'Ensure the object has the attribute you\'re trying to access',
                        'code_template': '# Check object type and available attributes'
                    }
                ]
            },
            'default': [
                {
                    'description': 'Review your code for logical errors',
                    'code_template': '# Debug your code to identify the issue'
                }
            ]
        }
    
    def generate(self, preprocessed_data, error_type, context_analysis):
        """Generate solution recommendations based on the error analysis.
        
        Args:
            preprocessed_data: A dictionary containing preprocessed error message and code context.
            error_type: The classified error type.
            context_analysis: The results of the context analysis.
            
        Returns:
            A list of dictionaries containing solution recommendations.
        """
        error_message = preprocessed_data.get('error_message', '')
        code_context = preprocessed_data.get('code_context', '')
        root_cause = context_analysis.get('root_cause', '')
        matches = context_analysis.get('matches', {})
        
        # Get the solution templates for the error type
        error_solutions = self.solution_templates.get(error_type, self.solution_templates['default'])
        
        # Determine the specific issue based on the root cause
        issue_type = self._determine_issue_type(root_cause)
        
        # Get solutions for the specific issue, or use default solutions for the error type
        solutions = error_solutions.get(issue_type, error_solutions.get('default', []))
        
        # Customize the solutions based on the code context and matches
        customized_solutions = self._customize_solutions(solutions, code_context, matches, error_type, root_cause)
        
        return customized_solutions
    
    def _determine_issue_type(self, root_cause):
        """Determine the specific issue type based on the root cause.
        
        Args:
            root_cause: The root cause string from context analysis.
            
        Returns:
            A string representing the specific issue type.
        """
        # Map root cause descriptions to specific issue types
        if 'missing parenthesis' in root_cause.lower():
            return 'missing_parenthesis'
        elif 'missing bracket' in root_cause.lower():
            return 'missing_bracket'
        elif 'missing brace' in root_cause.lower():
            return 'missing_brace'
        elif 'missing colon' in root_cause.lower():
            return 'missing_colon'
        elif 'indentation' in root_cause.lower():
            return 'invalid_indentation'
        elif 'string as a number' in root_cause.lower():
            return 'string_as_number'
        elif 'none' in root_cause.lower():
            return 'none_operation'
        elif 'incorrect arguments' in root_cause.lower():
            return 'wrong_function_args'
        elif 'non-iterable' in root_cause.lower():
            return 'non_iterable'
        elif 'undefined variable' in root_cause.lower():
            return 'undefined_variable'
        elif 'misspelling' in root_cause.lower():
            return 'misspelled_variable'
        elif 'scope' in root_cause.lower():
            return 'wrong_scope'
        elif 'out of range' in root_cause.lower():
            return 'out_of_bounds'
        elif 'empty list' in root_cause.lower():
            return 'empty_list'
        elif 'loop' in root_cause.lower():
            return 'wrong_loop_condition'
        elif 'key' in root_cause.lower() and 'exist' in root_cause.lower():
            return 'missing_key'
        elif 'key' in root_cause.lower() and 'type' in root_cause.lower():
            return 'wrong_key_type'
        elif 'dividing by zero' in root_cause.lower():
            return 'explicit_zero_division'
        elif 'variable' in root_cause.lower() and 'zero' in root_cause.lower():
            return 'variable_zero_division'
        elif 'attribute' in root_cause.lower() and 'exist' in root_cause.lower():
            return 'undefined_attribute'
        elif 'attribute' in root_cause.lower() and 'none' in root_cause.lower():
            return 'none_attribute'
        
        # Default to 'default' if no specific issue type is identified
        return 'default'
    
    def _customize_solutions(self, solutions, code_context, matches, error_type, root_cause):
        """Customize solution templates based on the specific code context.
        
        Args:
            solutions: List of solution template dictionaries.
            code_context: The code context string.
            matches: Dictionary of pattern matches from context analysis.
            error_type: The classified error type.
            root_cause: The root cause string from context analysis.
            
        Returns:
            A list of customized solution dictionaries.
        """
        customized_solutions = []
        
        for solution in solutions:
            # Create a copy of the solution template
            custom_solution = {
                'description': solution['description'],
                'code': solution.get('code_template', '')
            }
            
            # Extract variables from the code context based on the error type and matches
            variables = self._extract_variables(code_context, matches, error_type, root_cause)
            
            # Replace placeholders in the code template with actual values
            if variables:
                for var_name, var_value in variables.items():
                    placeholder = '{' + var_name + '}'
                    if placeholder in custom_solution['code']:
                        custom_solution['code'] = custom_solution['code'].replace(placeholder, var_value)
            
            # Add the customized solution to the list
            customized_solutions.append(custom_solution)
        
        return customized_solutions
    
    def _extract_variables(self, code_context, matches, error_type, root_cause):
        """Extract variables from the code context for solution customization.
        
        Args:
            code_context: The code context string.
            matches: Dictionary of pattern matches from context analysis.
            error_type: The classified error type.
            root_cause: The root cause string from context analysis.
            
        Returns:
            A dictionary of variable names and values extracted from the code context.
        """
        variables = {}
        
        # Extract variables based on the error type and matches
        if error_type == 'syntax_error':
            if 'missing_parenthesis' in matches and matches['missing_parenthesis']:
                code_snippet = matches['missing_parenthesis'][0]
                variables['code_snippet'] = code_snippet
                # Try to fix the code by adding the missing parenthesis
                if code_snippet.endswith('('):
                    variables['fixed_code'] = code_snippet + ')'
                elif code_snippet.startswith(')'):
                    variables['fixed_code'] = '(' + code_snippet
            
            # Similar logic for other syntax error types...
        
        elif error_type == 'name_error':
            if 'undefined_variable' in matches and matches['undefined_variable']:
                for match in matches['undefined_variable']:
                    if isinstance(match, tuple) and len(match) > 0:
                        variables['variable_name'] = match[0]
                    elif isinstance(match, str):
                        variables['variable_name'] = match
        
        elif error_type == 'index_error':
            if 'out_of_bounds' in matches and matches['out_of_bounds']:
                match = matches['out_of_bounds'][0]
                # Extract list name and index from something like "list_name[5]"
                list_match = re.match(r'(\w+)\s*\[\s*(\d+)\s*\]', match)
                if list_match:
                    variables['list_name'] = list_match.group(1)
                    variables['index'] = list_match.group(2)
        
        elif error_type == 'key_error':
            if 'missing_key' in matches and matches['missing_key']:
                match = matches['missing_key'][0]
                # Extract dictionary name and key from something like "dict_name['key']"
                dict_match = re.match(r'(\w+)\s*\[\s*["\'](\w+)["\']\s*\]', match)
                if dict_match:
                    variables['dict_name'] = dict_match.group(1)
                    variables['key'] = dict_match.group(2)
        
        elif error_type == 'division_by_zero':
            if 'variable_zero_division' in matches and matches['variable_zero_division']:
                match = matches['variable_zero_division'][0]
                # Extract dividend and divisor from something like "x / y"
                div_match = re.match(r'\s*(\w+)\s*\/\s*(\w+)', match)
                if div_match:
                    variables['dividend'] = div_match.group(1)
                    variables['divisor'] = div_match.group(2)
        
        elif error_type == 'attribute_error':
            if 'undefined_attribute' in matches and matches['undefined_attribute']:
                match = matches['undefined_attribute'][0]
                # Extract object and attribute from something like "obj.attr"
                attr_match = re.match(r'(\w+)\s*\.\s*(\w+)', match)
                if attr_match:
                    variables['object'] = attr_match.group(1)
                    variables['attribute'] = attr_match.group(2)
        
        return variables