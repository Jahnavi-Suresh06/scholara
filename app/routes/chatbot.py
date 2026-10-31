"""
app/routes/chatbot.py

ResearchBot Controller.
Intelligently classifies user queries, routes them to the appropriate response pipeline,
and provides structured responses for:
1. GENERAL_KNOWLEDGE: Clear concept explanations with examples and formatting, NO paper dumps.
2. RESEARCH_PAPER_SEARCH: Accurate semantic paper retrieval ranked by SBERT similarity with paper cards.
3. MIXED_QUERY: Clear concept explanation followed by relevant research paper cards.
4. PAPER_SUMMARIZATION: Factual, structured summary of a research paper's methodology and findings.
"""

from flask import Blueprint, render_template, request, jsonify, current_app

from app import db
from app.models.paper import Paper
from app.services.query_classifier import classify_query, IntentType
from app.services.answer_generator import generate_answer
from app.services.paper_retrieval_service import retrieve_papers_for_query, _format_paper


chatbot_bp = Blueprint("chatbot", __name__)


@chatbot_bp.route("/chatbot")
def chatbot():
    """Renders the ResearchBot interface."""
    return render_template("chatbot.html")


@chatbot_bp.route("/api/chatbot", methods=["POST"])
def chatbot_api():
    """
    Intelligent query processing API for ResearchBot.
    Accepts JSON:
    {
        "message": "User query string",
        "history": [{"role": "user"|"bot", "text": "...", "papers": [...]}]
    }
    """
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({
            "success": False,
            "error": "Please enter a question or research topic."
        }), 400

    try:
        # 1. Analyze and classify user intent
        analysis = classify_query(message, history=history)
        intent = analysis.intent
        topic = analysis.topic

        current_app.logger.info(
            f"ResearchBot query: {message!r} -> Intent: {intent}, Topic: {topic!r}, Count: {analysis.paper_count}"
        )

        # 2. Process according to classified intent

        # =========================================================================
        # CASE A: GENERAL KNOWLEDGE QUESTION
        # Examples: "What is Machine Learning?", "Explain Deep Learning", "What is NLP?",
        # "What are the applications of AI?", "What is the difference between AI and ML?"
        # =========================================================================
        if intent == IntentType.GENERAL_KNOWLEDGE:
            answer = generate_answer(analysis, history=history)
            return jsonify({
                "success": True,
                "intent": intent,
                "topic": topic,
                "answer": answer,
                "papers": []  # Never display paper recommendations for ordinary knowledge questions
            })

        # =========================================================================
        # CASE B: RESEARCH PAPER SEARCH
        # Examples: "Give me five research papers on Machine Learning",
        # "Find research papers related to CNN", "Show me papers about sentiment analysis"
        # =========================================================================
        elif intent == IntentType.RESEARCH_PAPER_SEARCH:
            retrieval = retrieve_papers_for_query(
                topic,
                top_k=analysis.paper_count
            )
            return jsonify({
                "success": True,
                "intent": intent,
                "topic": topic,
                "answer": retrieval["message"],
                "papers": retrieval["papers"]
            })

        # =========================================================================
        # CASE C: MIXED QUERY (Explanation + Relevant Research Papers)
        # Examples: "Explain Machine Learning and give me five related papers",
        # "What is Deep Learning? Also give me five research papers"
        # =========================================================================
        elif intent == IntentType.MIXED_QUERY:
            # 1. Generate thorough concept explanation
            explanation = generate_answer(analysis, history=history)

            # 2. Retrieve relevant research papers
            retrieval = retrieve_papers_for_query(
                topic,
                top_k=analysis.paper_count
            )

            # Both sections must be included
            combined_answer = (
                f"{explanation}\n\n"
                f"---\n\n"
                f"### Relevant Research Publications\n"
                f"{retrieval['message']}"
            )

            return jsonify({
                "success": True,
                "intent": intent,
                "topic": topic,
                "answer": combined_answer,
                "papers": retrieval["papers"]
            })

        # =========================================================================
        # CASE D: RESEARCH PAPER SUMMARIZATION
        # Examples: "Summarize this research paper", "Explain the abstract of this paper",
        # "What is the methodology used in this research?"
        # =========================================================================
        elif intent == IntentType.PAPER_SUMMARIZATION:
            # If no paper ID was resolved by classifier, search for top matching paper
            target_paper = None
            if analysis.paper_id:
                target_paper = db.session.get(Paper, analysis.paper_id)
            elif analysis.paper_title:
                target_paper = Paper.query.filter(Paper.title.ilike(f"%{analysis.paper_title}%")).first()

            if not target_paper and topic:
                # Search for the best paper on the topic
                retrieval = retrieve_papers_for_query(topic, top_k=1)
                if retrieval["papers"]:
                    best_id = retrieval["papers"][0]["id"]
                    target_paper = db.session.get(Paper, best_id)
                    analysis.paper_id = best_id

            if target_paper:
                summary_answer = generate_answer(analysis, history=history)
                paper_card = [_format_paper(target_paper, score=1.0)]
                return jsonify({
                    "success": True,
                    "intent": intent,
                    "topic": target_paper.title,
                    "answer": summary_answer,
                    "papers": paper_card
                })
            else:
                fallback_msg = (
                    "Please specify which research paper you would like me to summarize, "
                    "or search for papers first (e.g. *'Find research papers on Deep Learning'*), "
                    "and then click **Summarize** on any paper card."
                )
                return jsonify({
                    "success": True,
                    "intent": intent,
                    "topic": topic,
                    "answer": fallback_msg,
                    "papers": []
                })

        # Fallback safeguard
        answer = generate_answer(analysis, history=history)
        return jsonify({
            "success": True,
            "intent": IntentType.GENERAL_KNOWLEDGE,
            "topic": topic,
            "answer": answer,
            "papers": []
        })

    except Exception as e:
        current_app.logger.exception(f"Error in ResearchBot API: {e}")
        return jsonify({
            "success": False,
            "error": "An error occurred while processing your query. Please try again."
        }), 500
