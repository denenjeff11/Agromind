import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Force load environment variables
load_dotenv()

# Initialize the Gemini client
client = genai.Client()

SYSTEM_PROMPT = """You be an agric expert wey sabi Nigerian farmers well well.
You dey explain plant disease and treatment for simple Nigerian Pidgin English,
short and clear, no long grammar. Be warm and respectful, like you dey talk to
your own family farmer. Only recommend chemicals/treatment that dey inside the
data wey dem give you - no add your own chemical suggestion. If farmer ask
question wey no dey inside the data, tell am make e ask agro-dealer or extension
officer for more help."""

def explain_diagnosis(diagnosis: dict, treatment: dict) -> str:
    """First message to the farmer: explains what's wrong and what to do."""
    user_prompt = f"""
Disease detected: {treatment['name']}
Confidence: {diagnosis['confidence'] * 100:.0f}%
Cause: {treatment['cause']}
Symptoms: {treatment['symptoms']}
Recommended treatment: {treatment['treatment']}
Chemicals to use: {', '.join(treatment['chemicals'])}

Explain this to the farmer for Pidgin. Tell am wetin dey worry the plant,
why e happen, and wetin e go do to solve am. End by asking if e get any
question, or how many plants dey affected.
"""
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.5,
            )
        )
        return response.text
    except Exception as e:
        return f"Error with Gemini AI layer: {str(e)}"

def continue_conversation(diagnosis: dict, treatment: dict, chat_history: list, farmer_message: str) -> str:
    pass
