import os
import re
import asyncio
import threading
import time
from groq import Groq
from dotenv import load_dotenv
from typing import Tuple, Optional
import concurrent.futures
import sys

# Load environment variables
load_dotenv()

class CodeBERTModel:
    """A code error fixing model using Groq's LLaMA-3.3-70B-versatile model."""
    
    def __init__(self):
        """Initialize the model with LLaMA-3.3-70B-versatile."""
        self.model_name = "llama-3.3-70b-versatile"
        self.client = None  # Client will be initialized lazily
        self._model_loaded = False
        self._loading_lock = threading.Lock()
        
        self.error_types = [
            'syntax_error',
            'type_error',
            'name_error',
            'index_error',
            'key_error',
            'value_error',
            'attribute_error',
            'import_error',
            'division_by_zero',
            'indentation_error',
            'unknown_error'
        ]
        
        self.error_patterns = {
            'syntax_error': [
                r'SyntaxError', r'invalid syntax', r'unexpected token', r'EOF while scanning',
                r'expected.*:', r'missing.*\)', r'unmatched.*\(', r'invalid character',
                r'unexpected EOF', r'unexpected indent', r'invalid decimal literal'
            ],
            'type_error': [
                r'TypeError', r'not callable', r'unsupported operand', r'must be.*not',
                r'can\'t convert', r'argument must be', r'takes.*positional argument',
                r'missing.*required.*argument', r'unexpected keyword argument'
            ],
            'name_error': [
                r'NameError', r'not defined', r'referenced before assignment',
                r'global name.*not defined', r'local variable.*referenced'
            ],
            'index_error': [
                r'IndexError', r'out of range', r'list index out of range',
                r'string index out of range', r'tuple index out of range'
            ],
            'key_error': [r'KeyError', r'key.*not found', r'dictionary.*key'],
            'value_error': [
                r'ValueError', r'invalid literal', r'could not convert',
                r'not enough values', r'too many values', r'invalid value'
            ],
            'attribute_error': [
                r'AttributeError', r'has no attribute', r'object has no attribute',
                r'module.*has no attribute', r'type object.*has no attribute'
            ],
            'import_error': [
                r'ImportError', r'ModuleNotFoundError', r'No module named',
                r'cannot import', r'attempted relative import'
            ],
            'division_by_zero': [
                r'ZeroDivisionError', r'division by zero', r'float division by zero',
                r'integer division.*by zero'
            ],
            'indentation_error': [
                r'IndentationError', r'expected an indented block',
                r'unindent does not match', r'unexpected indent'
            ]
        }
        
        self.fix_patterns = {
            'syntax_error': {
                'missing_colon': (r'(if|elif|else|for|while|def|class|try|except|finally|with)\s+[^:]*$', r'\1 \2:'),
                'missing_parentheses': (r'print\s+([^\(].*)', r'print(\1)'),
                'missing_quotes': (r'([^"\'])([a-zA-Z_][a-zA-Z0-9_]*)([^"\'])', r'\1"\2"\3')
            },
            'indentation_error': {
                'fix_indent': (r'^(\s*)(.*)', r'    \2')  # Add 4 spaces
            },
            'name_error': {
                'common_typos': {
                    'lenght': 'length', 'pirnt': 'print', 'retrun': 'return',
                    'improt': 'import', 'fro ': 'for ', 'whiel': 'while'
                }
            }
        }
        
        self.is_initialized = False
        self._thread_local = threading.local()

    def _ensure_model_loaded(self):
        """Ensure the Groq model is loaded using lazy loading with thread safety."""
        if self._model_loaded:
            return True
        
        with self._loading_lock:
            if self._model_loaded:
                return True
                
            try:
                sys.stderr.write("\n*** Loading Groq model... ***\n")
                sys.stderr.flush()
                
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set in .env file")
                
                # Initialize Groq client per thread
                if not hasattr(self._thread_local, 'client'):
                    self._thread_local.client = Groq(api_key=api_key)
                
                self.client = self._thread_local.client
                
                # Test connection
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert Python code debugger."},
                            {"role": "user", "content": "Hello"}
                        ],
                        temperature=0.3,
                        max_tokens=10
                    )
                    if response.choices[0].message.content:
                        sys.stderr.write("\n*** Groq connection test successful ***\n")
                except Exception as e:
                    sys.stderr.write(f"\n*** Groq test failed: {e} ***\n")
                
                self._model_loaded = True
                self.is_initialized = True
                sys.stderr.write("\n*** Groq model loaded successfully ***\n")
                return True
                
            except Exception as e:
                sys.stderr.write(f"\n*** Error loading Groq model: {e} ***\n")
                sys.stderr.write("\nPlease ensure you have:\n")
                sys.stderr.write("1. Set GROQ_API_KEY in your .env file\n")
                sys.stderr.write("2. Installed groq package: pip install groq\n")
                sys.stderr.flush()
                self._model_loaded = False
                self.is_initialized = False
                return False
    
    def initialize(self):
        """Legacy method for backward compatibility."""
        return self._ensure_model_loaded()
    
    def classify(self, code, error_message):
        """Classify the type of error using enhanced rule-based patterns."""
        error_type = self._rule_based_classify(error_message)
        confidence = self._calculate_confidence(error_message, error_type)
        suggested_fix = self._get_rule_based_fix(code, error_type, error_message)
        
        print(f"Classification: {error_type} (confidence: {confidence:.2f})")
        return error_type, confidence, suggested_fix
    
    def _rule_based_classify(self, error_message):
        """Classify the error type using rule-based patterns."""
        if not error_message:
            return 'unknown_error'
            
        error_message = error_message.lower()
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern.lower(), error_message):
                    return error_type
        
        return 'unknown_error'
    
    def _calculate_confidence(self, error_message, error_type):
        """Calculate confidence score for the classification."""
        if error_type == 'unknown_error':
            return 0.1
            
        if not error_message:
            return 0.3
            
        error_message = error_message.lower()
        patterns = self.error_patterns.get(error_type, [])
        
        matches = 0
        for pattern in patterns:
            if re.search(pattern.lower(), error_message):
                matches += 1
                
        confidence = min(0.9, 0.5 + (matches * 0.1))
        return confidence
    
    def _get_rule_based_fix(self, code, error_type, error_message):
        """Generate a rule-based fix for common errors."""
        try:
            if error_type == 'syntax_error':
                return self._fix_syntax_error(code, error_message)
            elif error_type == 'indentation_error':
                return self._fix_indentation_error(code)
            elif error_type == 'name_error':
                return self._fix_name_error(code, error_message)
            elif error_type == 'type_error':
                return self._fix_type_error(code, error_message)
        except Exception as e:
            print(f"Error in rule-based fix: {e}")
            
        return None
    
    def _fix_syntax_error(self, code, error_message):
        """Fix common syntax errors."""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            
            if re.search(r'(if|elif|else|for|while|def|class|try|except|finally|with)\s+[^:]*$', line.strip()):
                if not line.strip().endswith(':'):
                    fixed_line = line.rstrip() + ':'
            
            if re.search(r'print\s+[^\(]', line):
                fixed_line = re.sub(r'print\s+([^\(].*)', r'print(\1)', line)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_indentation_error(self, code):
        """Fix indentation errors by standardizing to 4 spaces."""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_level = leading_spaces // 4
                fixed_line = '    ' * indent_level + line.lstrip()
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_name_error(self, code, error_message):
        """Fix common name errors and typos."""
        fixed_code = code
        
        typos = self.fix_patterns['name_error']['common_typos']
        for typo, correction in typos.items():
            fixed_code = re.sub(r'\b' + typo + r'\b', correction, fixed_code)
        
        return fixed_code
    
    def _fix_type_error(self, code, error_message):
        """Fix common type errors."""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            fixed_line = line
            
            if 'can\'t convert' in error_message.lower() or 'unsupported operand' in error_message.lower():
                fixed_line = re.sub(r'(\w+)\s*\+\s*(\w+)', r'str(\1) + str(\2)', line)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _run_async_fix(self, code, error_type, error_message, error_line=None):
        """Run the async fix generation in a clean event loop."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self._generate_fix_async(code, error_type, error_message, error_line)
            )
        finally:
            if hasattr(loop, 'is_closed') and not loop.is_closed():
                loop.close()

    async def _generate_fix_async(self, code, error_type, error_message, error_line=None):
        """Generate a fix using LLaMA-3.3-70B-versatile model asynchronously."""
        if not self.is_initialized:
            sys.stderr.write("\n*** Groq model not initialized. Cannot generate fix. ***\n")
            sys.stderr.flush()
            return code
        
        try:
            problematic_types = ['syntax_error', 'type_error', 'name_error', 'reference_error']
            is_problematic = error_type.lower() in problematic_types
            
            prompt = self._create_enhanced_prompt(code, error_type, error_message, error_line) if is_problematic else self._create_gemini_prompt(code, error_type, error_message, error_line)
            
            sys.stderr.write(f"\n*** Generated prompt for Groq: {prompt[:200]}... ***\n")
            sys.stderr.flush()
            
            try:
                client = self._thread_local.client
                
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert Python code debugger."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1000,
                    top_p=0.8
                )
                
                response_text = response.choices[0].message.content
                
                sys.stderr.write(f"\n*** Raw Groq response: {response_text[:300]}... ***\n")
                sys.stderr.flush()
                
                fixed_code = self._extract_gemini_fix(response_text, code)
                
                if not fixed_code or len(fixed_code.strip()) < 10:
                    sys.stderr.write("\n*** Warning: Extracted fix is too short or empty. Using original code. ***\n")
                    sys.stderr.flush()
                    return code
                
                if is_problematic:
                    if error_type.lower() == 'syntax_error' and 'expected' in error_message.lower() and ':' not in fixed_code:
                        sys.stderr.write("\n*** Warning: Fix doesn't address syntax error (missing colon). Using rule-based fix. ***\n")
                        sys.stderr.flush()
                        return self._get_rule_based_fix(code, error_type, error_message) or code
                    elif error_type.lower() == 'name_error' and 'not defined' in error_message.lower():
                        undefined_var = re.search(r"'([^']+)'\s+is not defined", error_message)
                        if undefined_var and undefined_var.group(1) not in fixed_code:
                            sys.stderr.write(f"\n*** Warning: Fix doesn't address undefined variable {undefined_var.group(1)}. Using rule-based fix. ***\n")
                            sys.stderr.flush()
                            return self._get_rule_based_fix(code, error_type, error_message) or code
                
                sys.stderr.write(f"\n*** Final fixed code: {fixed_code[:200]}... ***\n")
                sys.stderr.flush()
                return fixed_code
                
            except Exception as api_error:
                sys.stderr.write(f"\n*** Error calling Groq API: {api_error} ***\n")
                sys.stderr.write(f"\n{traceback.format_exc()}\n")
                sys.stderr.flush()
                
                simple_prompt = f"You are a Python expert. Fix this {error_type} in the code below. Return ONLY the fixed code with no explanations.\n\nError: {error_message}\n\nCode to fix:\n```python\n{code}\n```\n\nFixed code:"
                
                try:
                    sys.stderr.write("\n*** Retrying with simpler prompt... ***\n")
                    sys.stderr.flush()
                    
                    retry_response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a Python expert."},
                            {"role": "user", "content": simple_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=1000,
                        top_p=0.8
                    )
                    
                    retry_text = retry_response.choices[0].message.content
                    sys.stderr.write(f"\n*** Retry response: {retry_text[:300]}... ***\n")
                    sys.stderr.flush()
                    
                    fixed_code = self._extract_gemini_fix(retry_text, code)
                    if fixed_code and len(fixed_code.strip()) >= 10:
                        return fixed_code
                except Exception as retry_error:
                    sys.stderr.write(f"\n*** Retry also failed: {retry_error} ***\n")
                    sys.stderr.flush()
                
                return self._get_rule_based_fix(code, error_type, error_message) or code
            
        except Exception as e:
            sys.stderr.write(f"\n*** Error generating fix with Groq: {e} ***\n")
            sys.stderr.write(f"\n{traceback.format_exc()}\n")
            sys.stderr.flush()
            return code
    
    def generate_fix(self, code, error_type, error_message, error_line=None):
        """Generate a fix for the given code error using LLaMA-3.3-70B-versatile."""
        max_retries = 2
        for attempt in range(max_retries + 1):
            if self._ensure_model_loaded():
                break
            if attempt < max_retries:
                sys.stderr.write(f"\n*** Retry {attempt+1}/{max_retries} loading Groq model... ***")
                sys.stderr.flush()
                time.sleep(1)
            else:
                sys.stderr.write("\n*** Failed to load Groq model after retries. Using rule-based fix as fallback. ***")
                sys.stderr.flush()
                return self._get_rule_based_fix(code, error_type, error_message) or code

        try:
            sys.stderr.write("\n*** Running Groq fix generation... ***\n")
            sys.stderr.flush()
            
            fixed_code = self._run_async_fix(code, error_type, error_message, error_line)
            return fixed_code

        except Exception as e:
            sys.stderr.write(f"\n*** Error in generate_fix: {e} ***\n")
            sys.stderr.write(traceback.format_exc() + "\n")
            sys.stderr.flush()

            return self._get_rule_based_fix(code, error_type, error_message) or code
    
    def _create_gemini_prompt(self, code, error_type, error_message=None, error_line=None):
        """Create a prompt for the Groq model to generate a fix."""
        prompt_parts = [
            "You are an expert Python code debugger.",
            "The following code contains errors that need to be fixed.",
            "Your task is to identify and resolve the issues in the code based on the error type and message provided.",
            "Return ONLY the corrected code without any explanations, markdown formatting, or additional text.",
            "Ensure the fixed code is different from the input code and addresses the specified error.",
            ""
        ]
        
        if error_type:
            prompt_parts.append(f"Error Type: {error_type}")
        
        if error_message:
            prompt_parts.append(f"Error Message: {error_message}")
        
        if error_line is not None:
            prompt_parts.append(f"Error Line: {error_line}")
        
        prompt_parts.extend([
            "",
            "Code to fix:",
            "```python",
            code.strip(),
            "```",
            "",
            "Fixed code:"
        ])
        
        return '\n'.join(prompt_parts)
        
    def _create_enhanced_prompt(self, code, error_type, error_message=None, error_line=None):
        """Create an enhanced prompt for problematic error types."""
        prompt_parts = [
            "You are an expert Python code debugger specializing in fixing common errors.",
            "Your task is to fix the following code error and return ONLY the corrected code.",
            "Do not include any explanations, comments, or markdown formatting in your response.",
            "Ensure the fixed code is different from the input code and directly addresses the specified error.",
            ""
        ]
        
        if error_type.lower() == 'syntax_error':
            prompt_parts.extend([
                "INSTRUCTIONS FOR SYNTAX ERRORS:",
                "1. Fix missing colons after if/else, loops, function/class definitions.",
                "2. Correct mismatched parentheses, brackets, or quotes.",
                "3. Ensure proper indentation (4 spaces).",
                "4. Fix unclosed strings or invalid syntax."
            ])
        elif error_type.lower() == 'type_error':
            prompt_parts.extend([
                "INSTRUCTIONS FOR TYPE ERRORS:",
                "1. Add type conversions (str(), int(), float()) for incompatible types.",
                "2. Ensure function arguments match expected types.",
                "3. Fix operations between incompatible types (e.g., string + integer)."
            ])
        elif error_type.lower() == 'name_error':
            prompt_parts.extend([
                "INSTRUCTIONS FOR NAME ERRORS:",
                "1. Define any undefined variables or fix typos.",
                "2. Ensure variables are assigned before use.",
                "3. Add missing import statements or variable declarations."
            ])
        elif error_type.lower() == 'reference_error':
            prompt_parts.extend([
                "INSTRUCTIONS FOR REFERENCE ERRORS:",
                "1. Define undefined variables or functions.",
                "2. Check for scope issues or missing declarations."
            ])
        
        prompt_parts.append("")
        if error_type:
            prompt_parts.append(f"Error Type: {error_type}")
        
        if error_message:
            prompt_parts.append(f"Error Message: {error_message}")
            if error_type.lower() == 'name_error' and 'is not defined' in error_message:
                var_match = re.search(r"'([^']+)'\s+is not defined", error_message)
                if var_match:
                    undefined_var = var_match.group(1)
                    prompt_parts.append(f"Undefined Variable: {undefined_var}")
                    prompt_parts.append(f"You MUST define or fix the variable '{undefined_var}'.")
            elif error_type.lower() == 'syntax_error' and 'expected' in error_message:
                prompt_parts.append("You MUST fix the syntax error as described in the error message.")
        
        if error_line is not None:
            prompt_parts.append(f"Error Line: {error_line}")
        
        prompt_parts.extend([
            "",
            "Code to fix:",
            "```python",
            code.strip(),
            "```",
            "",
            "Fixed code:"
        ])
        
        return '\n'.join(prompt_parts)
    
    def _extract_gemini_fix(self, response_text, original_code):
        """Extract the fixed code from Groq's response."""
        try:
            sys.stderr.write(f"\n*** Extracting fix from response: {response_text[:100]}... ***\n")
            sys.stderr.flush()
            
            fix_text = response_text.strip()
            
            prefixes_to_remove = [
                "Here's the fixed code:",
                "Here is the fixed code:",
                "Fixed code:",
                "The fixed code is:",
                "Fixed Python code:",
                "Corrected code:",
                "Here's the corrected code:",
                "Here is the corrected code:",
                "The corrected code is:"
            ]
            
            for prefix in prefixes_to_remove:
                if fix_text.lower().startswith(prefix.lower()):
                    fix_text = fix_text[len(prefix):].strip()
                    break
            
            code_block_pattern = r'```(?:python)?\s*\n(.+?)\n```'
            code_blocks = re.findall(code_block_pattern, fix_text, re.DOTALL)
            
            if code_blocks:
                fix_text = code_blocks[0].strip()
                sys.stderr.write(f"\n*** Extracted code block: {fix_text[:100]}... ***\n")
                sys.stderr.flush()
            else:
                fix_text = re.sub(r'^```python\s*|```$', '', fix_text)
                fix_text = re.sub(r'^```|```$', '', fix_text)
                sys.stderr.write(f"\n*** No code blocks found, cleaned text: {fix_text[:100]}... ***\n")
                sys.stderr.flush()
                
                markers = ['fixed code:', 'corrected code:']
                for marker in markers:
                    if marker in fix_text.lower():
                        parts = fix_text.lower().split(marker, 1)
                        if len(parts) > 1 and parts[1].strip():
                            fix_text = parts[1].strip()
                            sys.stderr.write(f"\n*** Extracted after marker '{marker}': {fix_text[:100]}... ***\n")
                            sys.stderr.flush()
                            break
            
            if not fix_text or len(fix_text) < 5:
                sys.stderr.write("\n*** Extracted fix is too short, returning original code ***\n")
                sys.stderr.flush()
                return original_code
            
            code_indicators = ['def ', 'class ', 'import ', 'print(', 'if ', 'for ', 'while ', '=', 'return ']
            has_code = any(indicator in fix_text for indicator in code_indicators)
            
            if not has_code:
                sys.stderr.write("\n*** Extracted text doesn't appear to contain code, returning original code ***\n")
                sys.stderr.flush()
                return original_code
                
            return fix_text
            
        except Exception as e:
            sys.stderr.write(f"Error extracting Groq fix: {e}\n")
            sys.stderr.flush()
            return original_code