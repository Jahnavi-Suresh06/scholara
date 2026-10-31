import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")
    from google import genai
    return genai.Client(api_key=api_key)


def ask_gemini(question, context=""):
    """
    Send a research-related question to Gemini.

    context can contain relevant research papers retrieved
    from our semantic search system.
    """
    client = get_gemini_client()

    prompt = f"""
You are ResearchBot, an AI research assistant for Scholara.
Your job is to help students and researchers understand research topics and papers.

Rules:
- Answer the user's question directly, starting with the actual answer in the very first sentence.
- Do NOT introduce yourself or greet the user.
- Do NOT open with a preamble such as "Based on the provided research context...". Go straight to the content.
- Use clear, academic explanations with bullet points and structured headings.
- If context is provided, draw upon it accurately without inventing facts.

Research Context:
{context}

User Question:
{question}
"""

    model_candidates = [MODEL_NAME, "gemini-1.5-flash", "gemini-2.0-flash"]
    last_err = None

    for m in model_candidates:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt
            )
            return response.text
        except Exception as e:
            last_err = e

    raise last_err
