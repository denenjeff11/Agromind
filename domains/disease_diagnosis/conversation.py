"""
LLM conversation layer using Gemini (new google-genai SDK).
Explains diagnosis + treatment in Pidgin.
Includes automatic retry for temporary 503 server overload.
"""
import os
import time
from google import genai
from google.genai import types
from google.genai import errors

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3.8-flash"

SYSTEM_PROMPT = """You be an agric expert wey sabi Nigerian farmers well well.
You dey explain plant disease and treatment for simple Nigerian Pidgin English,
short and clear, no long grammar. Be warm and respectful, like you dey talk to
your own family farmer. Only recommend chemicals/treatment that dey inside the
data wey dem give you - no add your own chemical suggestion."""


def _generate_with_retry(contents, config, max_retries=4):
    """Calls Gemini, automatically retrying on temporary 503 overload errors."""
    delay = 2
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
        except errors.ServerError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Gemini busy (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # back off longer each retry


def explain_diagnosis(diagnosis: dict, treatment: dict) -> str:
    if diagnosis.get("is_healthy"):
        user_prompt = f"""
The farmer's maize plant is healthy, no disease detected (confidence: {diagnosis['confidence'] * 100:.0f}%).
Tell the farmer for Pidgin say their plant dey fine, and give one or two tips to keep am healthy.
"""
    else:
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
    response = _generate_with_retry(
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


def continue_conversation(diagnosis: dict, treatment: dict, chat_history: list, farmer_message: str) -> str:
    context = f"""
Context - Disease: {treatment['name']}, Treatment: {treatment['treatment']},
Chemicals: {', '.join(treatment['chemicals'])}.
Only use this data for chemical advice.
"""
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT + context),
        history=chat_history,
    )

    delay = 2
    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = chat.send_message(farmer_message)
            return response.text
        except errors.ServerError:
            if attempt == max_retries - 1:
                raise
            print(f"Gemini busy (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
