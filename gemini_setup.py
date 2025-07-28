import os
from google import genai
from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
system_prompt = "You are an AI assistant that strictly answers based on the given context. " \
                "If the answer cannot be derived directly from the provided context, " \
                "respond with: 'I do not have enough information to answer that.'"
model = genai.GenerativeModel("gemini-2.0-flash")