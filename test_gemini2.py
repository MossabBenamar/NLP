from models.ml_model import CodeBERTModel
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyDbwGp8v2jy8hweBMCNzV8aptpL3E52ExE"

def test_gemini():  # Remove async
    model = CodeBERTModel()
    model.initialize()

    code = """
def calculate_sum(a, b)
    return a + b
"""
    error_message = "SyntaxError: expected ':' (<unknown>, line 2)"
    fixed_code = model.generate_fix(code, 'syntax_error', error_message)
    print("Fixed Code:\n", fixed_code)

if __name__ == "__main__":
    test_gemini()  # Remove asyncio.run()