import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key found: {api_key is not None}")

if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

# Initialize Gemini client
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1alpha")
)

# Test models
models_to_test = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-pro"]

for model_name in models_to_test:
    print(f"\nTesting model {model_name}...")
    try:
        # Test with a simple prompt
        prompt = "Say hello world"
        
        print(f"Attempting to generate content with {model_name}...")
        
        # Create a simple synchronous test
        response = None
        try:
            # Try using the models API with proper config
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_output_tokens=100
                )
            )
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error with generate_content: {e}")
            
            # Try with streaming API
            try:
                print("Trying streaming API instead...")
                response_text = ""
                for chunk in client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        top_p=0.8,
                        max_output_tokens=100
                    )
                ):
                    if hasattr(chunk, 'text'):
                        response_text += chunk.text
                print(f"Streaming response: {response_text}")
                response = response_text
            except Exception as e2:
                print(f"Error with streaming API: {e2}")
                response = None
        
        if response:
            print(f"Successfully generated content with {model_name}")
            break
        else:
            print(f"Failed to generate content with {model_name}")
            
    except Exception as e:
        print(f"Error with model {model_name}: {e}")

if not response:
    print("\nAll models failed. Please check your API key and model availability.")