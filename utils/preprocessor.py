import re

class Preprocessor:
    """A class for preprocessing code and error messages before analysis."""
    
    def __init__(self):
        """Initialize the preprocessor with common error patterns."""
        # Common error message patterns for different programming languages
        self.error_patterns = {
            'python': {
                'syntax_error': r'SyntaxError:\s*(.+)',
                'type_error': r'TypeError:\s*(.+)',
                'name_error': r'NameError:\s*(.+)',
                'index_error': r'IndexError:\s*(.+)',
                'key_error': r'KeyError:\s*(.+)',
                'attribute_error': r'AttributeError:\s*(.+)',
                'zero_division_error': r'ZeroDivisionError:\s*(.+)',
                'import_error': r'ImportError:\s*(.+)',
                'value_error': r'ValueError:\s*(.+)',
                'indentation_error': r'IndentationError:\s*(.+)'
            },
            'javascript': {
                'syntax_error': r'SyntaxError:\s*(.+)',
                'type_error': r'TypeError:\s*(.+)',
                'reference_error': r'ReferenceError:\s*(.+)',
                'range_error': r'RangeError:\s*(.+)',
                'uri_error': r'URIError:\s*(.+)',
                'eval_error': r'EvalError:\s*(.+)',
                'internal_error': r'InternalError:\s*(.+)'
            },
            'java': {
                'null_pointer': r'NullPointerException',
                'class_not_found': r'ClassNotFoundException',
                'index_out_of_bounds': r'IndexOutOfBoundsException',
                'arithmetic_exception': r'ArithmeticException',
                'illegal_argument': r'IllegalArgumentException',
                'io_exception': r'IOException'
            },
            'cpp': {
                'segmentation_fault': r'Segmentation fault',
                'undefined_reference': r'undefined reference to',
                'bad_alloc': r'std::bad_alloc',
                'null_pointer': r'nullptr',
                'out_of_range': r'out of range'
            }
        }
        
        # Line number extraction patterns for different languages
        self.line_number_patterns = {
            'python': r'line\s+(\d+)',
            'javascript': r'line\s+(\d+)',
            'java': r'line\s+(\d+)',
            'cpp': r'line\s+(\d+)'
        }
    
    def preprocess(self, code, error_message, language='python'):
        """Preprocess the code and error message for analysis.
        
        Args:
            code: The code string to preprocess.
            error_message: The error message string to preprocess.
            language: The programming language of the code (default: 'python').
            
        Returns:
            A dictionary containing preprocessed data.
        """
        # Normalize the code (remove excessive whitespace, normalize line endings)
        normalized_code = self._normalize_code(code)
        
        # Extract relevant information from the error message
        error_info = self._extract_error_info(error_message, language)
        
        # Extract the code context around the error line if a line number is available
        code_context = self._extract_code_context(normalized_code, error_info.get('line_number'))
        
        # Combine all preprocessed data
        preprocessed_data = {
            'normalized_code': normalized_code,
            'error_message': error_message,
            'error_type': error_info.get('error_type'),
            'error_details': error_info.get('error_details'),
            'line_number': error_info.get('line_number'),
            'code_context': code_context
        }
        
        return preprocessed_data
    
    def _normalize_code(self, code):
        """Normalize the code by standardizing line endings and removing excessive whitespace.
        
        Args:
            code: The code string to normalize.
            
        Returns:
            The normalized code string.
        """
        # Replace all line endings with '\n'
        normalized = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing whitespace from each line
        normalized = '\n'.join(line.rstrip() for line in normalized.split('\n'))
        
        return normalized
    
    def _extract_error_info(self, error_message, language):
        """Extract relevant information from the error message.
        
        Args:
            error_message: The error message string.
            language: The programming language of the code.
            
        Returns:
            A dictionary containing extracted error information.
        """
        error_info = {
            'error_type': None,
            'error_details': None,
            'line_number': None
        }
        
        # Extract the error type and details
        if language in self.error_patterns:
            for error_type, pattern in self.error_patterns[language].items():
                match = re.search(pattern, error_message)
                if match:
                    error_info['error_type'] = error_type
                    if len(match.groups()) > 0:
                        error_info['error_details'] = match.group(1)
                    break
        
        # Extract the line number
        if language in self.line_number_patterns:
            line_match = re.search(self.line_number_patterns[language], error_message)
            if line_match:
                try:
                    error_info['line_number'] = int(line_match.group(1))
                except ValueError:
                    pass
        
        return error_info
    
    def _extract_code_context(self, code, line_number, context_lines=3):
        """Extract the code context around the error line.
        
        Args:
            code: The normalized code string.
            line_number: The line number where the error occurred (1-based).
            context_lines: The number of lines to include before and after the error line.
            
        Returns:
            A string containing the code context, or the entire code if no line number is available.
        """
        if line_number is None:
            return code
        
        lines = code.split('\n')
        
        # Adjust line_number to 0-based indexing
        line_index = line_number - 1
        
        # Determine the start and end lines for the context
        start_line = max(0, line_index - context_lines)
        end_line = min(len(lines), line_index + context_lines + 1)
        
        # Extract the context lines
        context_lines = lines[start_line:end_line]
        
        # Add line numbers to the context
        numbered_context = []
        for i, line in enumerate(context_lines, start=start_line + 1):
            prefix = '> ' if i == line_number else '  '
            numbered_context.append(f"{prefix}{i}: {line}")
        
        return '\n'.join(numbered_context)