import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
import numpy as np
import re
import os
import threading
import time

class CodeBERTModel:
    """A lightweight and efficient code error fixing model with rule-based classification and optimized generation."""
    
    def __init__(self):
        """Initialize the model with lazy loading and optimized settings."""
        # Use a smaller, faster model for better performance
        self.generation_model_name = "microsoft/DialoGPT-small"  # Smaller, faster alternative
        self.generation_tokenizer = None
        self.generation_model = None
        
        # Set cache directory to avoid permission issues
        self.cache_dir = os.path.join(os.getcwd(), "model_cache")
        
        # Lazy loading flag
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
        
        # Enhanced error patterns for better rule-based classification
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
        
        # Common fix patterns for rule-based fixes
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
        
        # Don't initialize models on creation - use lazy loading
        self.is_initialized = False
    
    def _ensure_model_loaded(self):
        """Ensure the model is loaded using lazy loading with thread safety."""
        if self._model_loaded:
            return True
            
        with self._loading_lock:
            if self._model_loaded:  # Double-check after acquiring lock
                return True
                
            try:
                print("Loading lightweight model for code generation...")
                # Create cache directory if it doesn't exist
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # Load a smaller, faster tokenizer and model
                self.generation_tokenizer = AutoTokenizer.from_pretrained(
                    self.generation_model_name,
                    cache_dir=self.cache_dir,
                    padding_side='left'
                )
                
                # Set padding token
                if self.generation_tokenizer.pad_token is None:
                    self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token
                
                # Load a smaller causal language model
                self.generation_model = AutoModelForCausalLM.from_pretrained(
                    self.generation_model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                self.generation_model.eval()
                self._model_loaded = True
                self.is_initialized = True
                print("Model loaded successfully")
                return True
                
            except Exception as e:
                print(f"Error loading model: {e}")
                self._model_loaded = False
                self.is_initialized = False
                return False
    
    def initialize(self):
        """Legacy method for backward compatibility."""
        return self._ensure_model_loaded()
    
    def classify(self, code, error_message):
        """Classify the type of error using enhanced rule-based patterns.
        
        Args:
            code: The code containing the error.
            error_message: The error message from the compiler/interpreter.
            
        Returns:
            A tuple of (error_type, confidence_score, suggested_fix) where suggested_fix is optional.
        """
        # Use enhanced rule-based classification
        error_type = self._rule_based_classify(error_message)
        confidence = self._calculate_confidence(error_message, error_type)
        suggested_fix = self._get_rule_based_fix(code, error_type, error_message)
        
        print(f"Classification: {error_type} (confidence: {confidence:.2f})")
        return error_type, confidence, suggested_fix
    
    def _rule_based_classify(self, error_message):
        """Classify the error type using rule-based patterns.
        
        Args:
            error_message: The error message string.
            
        Returns:
            A string representing the classified error type, or 'unknown_error' if no match.
        """
        if not error_message:
            return 'unknown_error'
            
        error_message = error_message.lower()
        
        # Check each error type's patterns
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern.lower(), error_message):
                    return error_type
        
        return 'unknown_error'
    
    def _calculate_confidence(self, error_message, error_type):
        """Calculate confidence score for the classification.
        
        Args:
            error_message: The error message string.
            error_type: The classified error type.
            
        Returns:
            A float between 0 and 1 representing confidence.
        """
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
                
        # Higher confidence for more pattern matches
        confidence = min(0.9, 0.5 + (matches * 0.1))
        return confidence
    
    def _get_rule_based_fix(self, code, error_type, error_message):
        """Generate a rule-based fix for common errors.
        
        Args:
            code: The original code.
            error_type: The classified error type.
            error_message: The error message.
            
        Returns:
            A suggested fix string or None if no rule-based fix available.
        """
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
            
            # Fix missing colons
            if re.search(r'(if|elif|else|for|while|def|class|try|except|finally|with)\s+[^:]*$', line.strip()):
                if not line.strip().endswith(':'):
                    fixed_line = line.rstrip() + ':'
            
            # Fix print statements (Python 2 to 3)
            if re.search(r'print\s+[^\(]', line):
                fixed_line = re.sub(r'print\s+([^\(].*)', r'print(\1)', line)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_indentation_error(self, code):
        """Fix indentation errors by standardizing to 4 spaces."""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if line.strip():  # Non-empty line
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip())
                # Standardize to multiples of 4
                indent_level = leading_spaces // 4
                fixed_line = '    ' * indent_level + line.lstrip()
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_name_error(self, code, error_message):
        """Fix common name errors and typos."""
        fixed_code = code
        
        # Fix common typos
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
            
            # Fix string concatenation with numbers
            if 'can\'t convert' in error_message.lower() or 'unsupported operand' in error_message.lower():
                # Add str() around variables that might need conversion
                fixed_line = re.sub(r'(\w+)\s*\+\s*(\w+)', r'str(\1) + str(\2)', line)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _get_rule_based_fix(self, code, error_type, error_message):
        """Generate a rule-based fix for common errors.
        
        Args:
            code: The original code.
            error_type: The classified error type.
            error_message: The error message.
            
        Returns:
            A suggested fix string or None if no rule-based fix available.
        """
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
            
            # Fix missing colons
            if re.search(r'(if|elif|else|for|while|def|class|try|except|finally|with)\s+[^:]*$', line.strip()):
                if not line.strip().endswith(':'):
                    fixed_line = line.rstrip() + ':'
            
            # Fix print statements (Python 2 to 3)
            if re.search(r'print\s+[^\(]', line):
                fixed_line = re.sub(r'print\s+([^\(].*)', r'print(\1)', line)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_indentation_error(self, code):
        """Fix indentation errors by standardizing to 4 spaces."""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if line.strip():  # Non-empty line
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip())
                # Standardize to multiples of 4
                indent_level = leading_spaces // 4
                fixed_line = '    ' * indent_level + line.lstrip()
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_name_error(self, code, error_message):
        """Fix common name errors and typos."""
        fixed_code = code
        
        # Fix common typos
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
            
            # Fix string concatenation with numbers
            if 'can\'t convert' in error_message.lower() or 'unsupported operand' in error_message.lower():
                # Add str() around variables that might need conversion
                fixed_line = re.sub(r'(\w+)\s*\+\s*(\w+)', r'str(\1) + str(\2)', line)
            
            fixed_lines.append(fixed_line)
        
        return '\n'.join(fixed_lines)
    
    def generate_fix(self, code, error_type, error_message, error_line=None):
        """Generate a fix for the given code error using CodeT5+.
        
        Args:
            code: The code containing the error.
            error_type: The type of error (from classification).
            error_message: The error message.
            error_line: The line number where the error occurred (optional).
            
        Returns:
            A string containing the fixed code, or the original code if fix generation fails.
        """
        if not self.is_initialized:
            print("CodeT5+ model not initialized. Cannot generate fix.")
            return code
        
        try:
            # Create a comprehensive prompt for the model
            prompt = self._create_fix_prompt(code, error_type, error_message, error_line)
            print(f"Generated prompt for CodeT5+: {prompt[:200]}...")  # Show first 200 chars
            
            # Tokenize the prompt with proper attention mask
            inputs = self.generation_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            )
            
            # Generate fix with optimized parameters for code generation
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=200,  # Reasonable limit for code fixes
                    num_beams=4,         # Beam search for quality
                    temperature=0.8,     # Balanced creativity
                    do_sample=True,      # Enable sampling
                    top_p=0.95,         # Nucleus sampling
                    top_k=50,           # Top-k sampling
                    pad_token_id=self.generation_tokenizer.pad_token_id,
                    eos_token_id=self.generation_tokenizer.eos_token_id,
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # Avoid repetition
                    repetition_penalty=1.1   # Slight penalty for repetition
                )
            
            # Decode the generated output
            generated_text = self.generation_tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            print(f"Raw generated text: {generated_text[:300]}...")  # Debug output
            
            # Extract the fixed code from the generated text
            fixed_code = self._extract_fix(generated_text, code)
            
            # Validate the fix
            if fixed_code.strip() == code.strip():
                print("Generated fix is identical to original. Trying alternative approach...")
                # Try with a simpler, more direct prompt
                simple_prompt = f"Fix this {error_type}: {code}"
                simple_inputs = self.generation_tokenizer(
                    simple_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                
                outputs = self.generation_model.generate(
                    **simple_inputs,
                    max_new_tokens=150,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.generation_tokenizer.pad_token_id,
                    eos_token_id=self.generation_tokenizer.eos_token_id
                )
                
                generated_text = self.generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                fixed_code = self._extract_fix(generated_text, code)
            
            print(f"Final fixed code: {fixed_code[:200]}...")  # Debug output
            return fixed_code
            
        except Exception as e:
            print(f"Error generating fix with CodeT5+: {e}")
            import traceback
            print(traceback.format_exc())
            return code
    
    def _create_fix_prompt(self, code, error_type, error_message=None, error_line=None):
        """Create a prompt for the CodeT5+ model to generate a fix.
        
        Args:
            code: The code containing the error.
            error_type: The type of error.
            error_message: The error message (optional).
            error_line: The line number of the error (optional).
            
        Returns:
            A formatted prompt string optimized for CodeT5+.
        """
        # Create a structured prompt optimized for CodeT5+ code generation
        prompt_parts = []
        
        # Start with task instruction
        prompt_parts.append("Fix the code error:")
        
        # Add error details
        if error_type:
            prompt_parts.append(f"Error: {error_type}")
        
        if error_message:
            prompt_parts.append(f"Message: {error_message}")
        
        if error_line is not None:
            prompt_parts.append(f"Line: {error_line}")
        
        # Add the code with clear delimiters
        prompt_parts.append("\nOriginal code:")
        prompt_parts.append(code.strip())
        
        # Clear instruction for output
        prompt_parts.append("\nFixed code:")
        
        return '\n'.join(prompt_parts)
    
    def _extract_fix(self, generated_text, original_code):
        """Extract the fixed code from the model's generated text.
        
        Args:
            generated_text: The text generated by the model.
            original_code: The original code for fallback.
            
        Returns:
            The extracted fixed code as a string.
        """
        try:
            # Clean the generated text
            fix_text = generated_text.strip()
            
            # Strategy 1: Look for "Fixed code:" marker and extract what follows
            if "Fixed code:" in fix_text:
                fix_text = fix_text.split("Fixed code:", 1)[1].strip()
            elif "fixed code:" in fix_text.lower():
                # Case insensitive search
                lower_text = fix_text.lower()
                marker_pos = lower_text.find("fixed code:")
                if marker_pos != -1:
                    fix_text = fix_text[marker_pos + len("fixed code:"):].strip()
            
            # Strategy 2: Handle code blocks
            if "```" in fix_text:
                # Extract content between code blocks
                parts = fix_text.split("```")
                if len(parts) >= 3:
                    # Take the content of the first code block
                    code_block = parts[1]
                    # Remove language identifier if present
                    lines = code_block.split('\n')
                    if lines and lines[0].strip().lower() in ['python', 'py', 'code']:
                        lines = lines[1:]
                    fix_text = '\n'.join(lines).strip()
            
            # Strategy 3: Remove line numbers if present
            lines = fix_text.split('\n')
            cleaned_lines = []
            import re
            for line in lines:
                # Remove line numbers at the beginning (e.g., "1: ", "2: ")
                cleaned_line = re.sub(r'^\s*\d+:\s*', '', line)
                cleaned_lines.append(cleaned_line)
            
            fix_text = '\n'.join(cleaned_lines)
            
            # Strategy 4: Remove explanatory text and keep only code
            lines = fix_text.split('\n')
            code_lines = []
            
            for line in lines:
                stripped = line.strip()
                # Skip empty lines and explanatory text
                if not stripped:
                    code_lines.append(line)  # Keep empty lines for code structure
                elif stripped.startswith(('#', '//', '/*', '*')):
                    code_lines.append(line)  # Keep comments
                elif any(keyword in stripped for keyword in [
                    'def ', 'class ', 'import ', 'from ', 'if ', 'elif ', 'else:', 
                    'for ', 'while ', 'try:', 'except', 'finally:', 'with ', 
                    'return', 'yield', 'break', 'continue', 'pass', 'raise',
                    '=', '+=', '-=', '*=', '/=', '(', ')', '[', ']', '{', '}'
                ]):
                    code_lines.append(line)  # Keep code lines
                elif not any(phrase in stripped.lower() for phrase in [
                    'here is', 'this is', 'the fix', 'explanation', 'note that',
                    'to fix', 'the error', 'corrected', 'fixed', 'solution'
                ]):
                    code_lines.append(line)  # Keep if not explanatory
            
            if code_lines:
                fix_text = '\n'.join(code_lines)
            
            # Final cleanup
            fix_text = fix_text.strip()
            
            # Remove any remaining markdown
            fix_text = re.sub(r'```\w*\n?', '', fix_text)
            fix_text = fix_text.replace('```', '').strip()
            
            # If the result is empty or too short, return original code
            if not fix_text or len(fix_text.strip()) < 5:
                print("Extracted fix is too short, returning original code")
                return original_code
            
            return fix_text
            
        except Exception as e:
            print(f"Error extracting fix: {e}")
            # Return the original code if extraction fails
            return original_code