from flask import Blueprint, render_template, request, jsonify

from app.services.gemini_service import ask_gemini
from app.services.recommendation_service import semantic_search

chatbot_bp = Blueprint("chatbot", __name__)


@chatbot_bp.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


@chatbot_bp.route("/api/chatbot", methods=["POST"])
def chatbot_api():
    data = request.get_json() or {}

    question = data.get("message", "").strip()

    if not question:
        return jsonify({
            "success": False,
            "error": "Please enter a question."
        }), 400

    try:
        # ---------------------------------------------------------
        # STEP 1: Search the research-paper database
        # ---------------------------------------------------------
        search_result = semantic_search(question, top_k=5)

        context_parts = []
        papers = []

        if search_result.get("available"):
            for result in search_result.get("results", []):
                paper = result["paper"]
                score = result["score"]

                papers.append({
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "score": round(float(score), 3)
                })

                context_parts.append(
                    f"""
Paper Title: {paper.title}
Authors: {paper.authors}
Year: {paper.publication_year}
Journal/Conference: {paper.journal_conference}
Keywords: {paper.keywords}
Abstract: {paper.abstract}
Semantic Similarity: {score:.3f}
"""
                )

        # ---------------------------------------------------------
        # STEP 2: Combine retrieved papers into Gemini context
        # ---------------------------------------------------------
        if context_parts:
            research_context = "\n".join(context_parts)
        else:
            research_context = (
                "No relevant research papers were found in the "
                "database for this question."
            )

        # ---------------------------------------------------------
        # STEP 3: Ask Gemini using the retrieved research context
        # ---------------------------------------------------------
        answer = ask_gemini(
            question,
            context=research_context
        )

        # ---------------------------------------------------------
        # STEP 4: Return answer + papers to frontend
        # ---------------------------------------------------------
        return jsonify({
            "success": True,
            "answer": answer,
            "papers": papers
        })

    except Exception as e:
        print("CHATBOT ERROR:", e)

        return jsonify({
            "success": False,
            "error": "Something went wrong while processing your question."
        }), 500