import os
from dotenv import load_dotenv
import openai

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

def get_openai_client():
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
