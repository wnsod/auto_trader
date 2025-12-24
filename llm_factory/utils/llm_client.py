import os
import openai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # print("⚠️ OPENAI_API_KEY not found in .env, using Mock mode.")
        print("OPENAI_API_KEY not found in .env, using Mock mode.")
        return None
    return openai.OpenAI(api_key=api_key)

