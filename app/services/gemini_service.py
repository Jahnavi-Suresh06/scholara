import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.6-flash"


def ask_gemini(question, context=""):
    """
    Send a research-related question to Gemini.

    context can contain relevant research papers retrieved
    from our semantic search system.
    """

    prompt = f"""
You are ResearchBot, an AI research assistant for an
Scholara
Your job is to help students and researchers understand
research topics and papers.

Rules:
- Answer the user's question directly, starting with the actual
  answer in the very first sentence.
- Do NOT introduce yourself or greet the user (e.g. "Hello! I am
  ResearchBot..."). The user already sees a one-time welcome
  message elsewhere in the app -- never repeat it here.
- Do NOT open with a preamble such as "Based on the provided
  research papers/context, here is a clear explanation..." or any
  similar throat-clearing sentence. Go straight to the content.
- Only mention that you're drawing on the provided research context
  when it's genuinely necessary for the user to know that (e.g. to
  explain why you can't fully answer, or to attribute a specific
  claim to a specific paper) -- never as a routine opening line.
- Prefer information from the provided research context when it's
  relevant to the question.
- Do not invent paper details.
- If the provided context does not contain enough information,
  say so plainly and concisely, without a generic disclaimer
  template.
- Use simple, natural explanations.
- Use headings, numbered lists, or bullet points when they genuinely
  make the answer clearer -- not as a rigid template for every reply.
- When discussing research papers, mention relevant paper titles
  only when it's useful to the user, not automatically in every
  answer.
- Vary your phrasing across responses -- avoid repeating the same
  stock sentences answer after answer.

Research Context:
{context}

User Question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text
