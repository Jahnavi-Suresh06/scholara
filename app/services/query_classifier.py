"""
query_classifier.py

Intelligent query classification for ResearchBot.
Categorizes user queries into one of four distinct intents:
1. GENERAL_KNOWLEDGE: Concept explanations, definitions, comparisons, working mechanisms.
2. RESEARCH_PAPER_SEARCH: Requests for academic/research papers, publications, or literature.
3. MIXED_QUERY: Asks for both a conceptual explanation AND relevant research papers.
4. PAPER_SUMMARIZATION: Requests to summarize, analyze methodology, or extract findings of a paper.

Also extracts:
- Core technical topic (cleaned of conversational boilerplate)
- Requested paper count (e.g., 'five papers' -> 5)
- Referenced paper identifier/title (if specified or inherited from context)
- Special question traits (comparisons, applications, mechanisms)
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


class IntentType:
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    RESEARCH_PAPER_SEARCH = "RESEARCH_PAPER_SEARCH"
    MIXED_QUERY = "MIXED_QUERY"
    PAPER_SUMMARIZATION = "PAPER_SUMMARIZATION"


@dataclass
class QueryAnalysis:
    raw_query: str
    intent: str
    topic: str
    paper_count: int = 5
    paper_id: Optional[int] = None
    paper_title: Optional[str] = None
    is_comparison: bool = False
    is_applications: bool = False
    is_follow_up: bool = False
    is_empty: bool = False
    concept_a: Optional[str] = None
    concept_b: Optional[str] = None


# Word to number mapping
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

# Regex to detect paper search requests
# Matches: "give me papers", "find research papers", "recommend five papers",
# "show papers about", "papers published on", "list of papers", etc.
_PAPER_SEARCH_PATTERN = re.compile(
    r"\b("
    r"(give|find|show|search|fetch|get|recommend|suggest|list|provide|display)\b.*?\b(papers?|publications?|articles?|literature|studies)\b"
    r"|(research\s+papers?|academic\s+papers?|scientific\s+papers?)"
    r"|(papers?|publications?|articles?)\s+(on|about|related\s+to|regarding|for|involving|published\s+on)"
    r"|(recent|top|relevant|latest)\s+(papers?|publications?|articles?)"
    r"|(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(research\s+)?(papers?|articles?|publications?)"
    r")\b",
    re.IGNORECASE
)

# Regex to detect paper summarization requests
# Matches: "summarize this paper", "explain the abstract", "methodology used in this research",
# "what are the main findings", "summarize paper 123", etc.
_SUMMARIZE_PATTERN = re.compile(
    r"\b("
    r"(summarize|summary|summarise)\b.*?\b(paper|research|study|abstract|article|document|work|findings|methodology)\b"
    r"|(explain|what\s+is)\s+(the\s+)?abstract\b"
    r"|what\s+(is|are)\s+(the\s+)?methodolog(y|ies)\b"
    r"|what\s+are\s+(the\s+)?(main\s+)?findings\b"
    r"|(key\s+contributions?|main\s+results?)\s+of\s+(this|the)\s+paper\b"
    r"|summarize\s+(this|it)\b"
    r")\b",
    re.IGNORECASE
)

# Regex to detect explanation requests
# Matches: "what is", "explain", "how does it work", "why", "describe", etc.
_EXPLANATION_PATTERN = re.compile(
    r"\b("
    r"what\s+is|what\s+are|what\s+does|how\s+does|how\s+do|how\s+is|why\s+is|why\s+do"
    r"|explain|describe|clarify|overview|definition|define|tell\s+me\s+about"
    r"|difference\s+between|compare|vs|versus"
    r"|applications?\s+of|uses?\s+of|use\s+cases?\s+of"
    r"|types?\s+of|categories?\s+of|how\s+to|how\s+it\s+works"
    r")\b",
    re.IGNORECASE
)

# Comparison patterns
_COMPARISON_PATTERN = re.compile(
    r"(?:difference\s+between|compare|distinguish\s+between)\s+([a-zA-Z0-9\s\-]+?)\s+(?:and|vs\.?|versus)\s+([a-zA-Z0-9\s\-]+?)(?:\?|$)",
    re.IGNORECASE
)

# Applications patterns
_APPLICATIONS_PATTERN = re.compile(
    r"\b(applications?|uses?|use\s+cases?|how\s+is\s+.*?used)\b",
    re.IGNORECASE
)

# Phrases that look like search or research but are actually domain concepts
# e.g., "how is it used in research?" refers to scientific research as an activity, not fetching papers
_RESEARCH_AS_DOMAIN = re.compile(
    r"\b(in\s+research|for\s+research|research\s+process|research\s+work|research\s+methodology)\b",
    re.IGNORECASE
)


def _extract_paper_count(text: str) -> int:
    """Extract requested number of papers (e.g., 'five papers' -> 5). Defaults to 5, max 10."""
    match = re.search(r"\b(\d+)\s*(?:research\s+)?(?:papers?|articles?|publications?)\b", text, re.IGNORECASE)
    if match:
        count = int(match.group(1))
        return max(1, min(count, 10))

    match_word = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:research\s+)?(?:papers?|articles?|publications?)\b",
        text,
        re.IGNORECASE
    )
    if match_word:
        word = match_word.group(1).lower()
        return _WORD_NUMBERS.get(word, 5)

    return 5


def _extract_topic(text: str, is_paper_search: bool = False, is_mixed: bool = False) -> str:
    """
    Cleans boilerplate commands from the query to isolate the underlying topic.
    e.g. 'Give me five research papers on Machine Learning' -> 'Machine Learning'
    e.g. 'Explain Machine Learning and give me five related papers' -> 'Machine Learning'
    """
    cleaned = text.strip()

    # If mixed query, extract the topic from the explanation portion
    if is_mixed:
        # e.g. "Explain Machine Learning and suggest related research papers"
        m = re.match(
            r"^(?:explain|what\s+is|describe)\s+([a-zA-Z0-9\s\-]+?)(?:\s+and\s+|\s*\.\s*|\s*,\s*|\s+also\s+).*$",
            cleaned,
            re.IGNORECASE
        )
        if m:
            topic = m.group(1).strip()
            topic = re.sub(r"\b(and|also|suggest|give|show|papers?)\b.*$", "", topic, flags=re.IGNORECASE).strip()
            if topic:
                return topic

    # Strip common leading search commands
    prefixes_to_strip = [
        r"^(?:please\s+)?(?:give\s+me|find|show\s+me|show|search\s+for|search|fetch|get|recommend|suggest|list|display)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|some|a\s+few|top|recent|latest)?\s*(?:research\s+|academic\s+|scientific\s+)?(?:papers?|publications?|articles?|literature|studies)\s+(?:on|about|related\s+to|regarding|for|involving|published\s+on)\s+",
        r"^(?:give\s+me|find|show\s+me|show|recommend)\s+(?:research\s+)?papers?\s+(?:on|about|related\s+to|for)\s+",
        r"^(?:papers?|publications?|articles?)\s+(?:related\s+to|about|on|regarding)\s+",
        r"^(?:what\s+are\s+the\s+applications\s+of|applications\s+of|uses\s+of)\s+",
        r"^(?:what\s+is\s+the\s+difference\s+between|difference\s+between|compare)\s+",
        r"^(?:how\s+does\s+a\s+|how\s+does\s+an\s+|how\s+does\s+|how\s+do\s+)",
        r"^(?:what\s+is\s+a\s+|what\s+is\s+an\s+|what\s+is|what\s+are|explain|describe|clarify|overview\s+of|tell\s+me\s+about)\s+",
    ]

    for pat in prefixes_to_strip:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing fluff
    cleaned = re.sub(r"\s+work\??$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+and\s+(?:give|suggest|show|find|provide)\s+.*papers?.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+(?:please|thanks|thank\s+you)$", "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned if cleaned else text.strip()


def _extract_paper_reference(text: str, history: Optional[List[Dict[str, Any]]] = None) -> (Optional[int], Optional[str]):
    """
    Extracts an explicit paper ID or title from the query or recent chat history.
    """
    # Check for explicit ID: "paper 123", "id 456", "#123"
    id_match = re.search(r"\b(?:paper\s+(?:id\s+)?|id\s+|#)(\d+)\b", text, re.IGNORECASE)
    if id_match:
        return int(id_match.group(1)), None

    # Check for quoted title: "Summarize the paper 'Deep Learning'"
    title_match = re.search(r"['\"]([^'\"]{4,100})['\"]", text)
    if title_match:
        return None, title_match.group(1).strip()

    # If referring to "this paper" or "the paper", inspect recent history
    if history:
        for entry in reversed(history):
            # Check if previous bot response contained papers
            papers = entry.get("papers") or []
            if papers and len(papers) > 0:
                first_paper = papers[0]
                return first_paper.get("id"), first_paper.get("title")

    return None, None


def classify_query(query: str, history: Optional[List[Dict[str, Any]]] = None) -> QueryAnalysis:
    """
    Classifies a user query into GENERAL_KNOWLEDGE, RESEARCH_PAPER_SEARCH,
    MIXED_QUERY, or PAPER_SUMMARIZATION with extracted topic and parameters.
    """
    raw = (query or "").strip()
    if not raw:
        return QueryAnalysis(
            raw_query=raw,
            intent=IntentType.GENERAL_KNOWLEDGE,
            topic="",
            is_empty=True
        )

    # 1. Check for Comparison Intent
    comp_match = _COMPARISON_PATTERN.search(raw)
    is_comparison = bool(comp_match) or bool(re.search(r"\b(vs\.?|versus)\b", raw, re.IGNORECASE))
    concept_a, concept_b = None, None
    if comp_match:
        concept_a = comp_match.group(1).strip()
        concept_b = comp_match.group(2).strip()

    # 2. Check for Applications Intent
    is_applications = bool(_APPLICATIONS_PATTERN.search(raw))

    # 3. Check for Summarization Intent
    has_summarize = bool(_SUMMARIZE_PATTERN.search(raw))
    if has_summarize:
        paper_id, paper_title = _extract_paper_reference(raw, history)
        topic = paper_title or _extract_topic(raw)
        return QueryAnalysis(
            raw_query=raw,
            intent=IntentType.PAPER_SUMMARIZATION,
            topic=topic,
            paper_id=paper_id,
            paper_title=paper_title,
            is_comparison=is_comparison,
            is_applications=is_applications
        )

    # 4. Check for Paper Search and Explanation flags
    has_paper_search = bool(_PAPER_SEARCH_PATTERN.search(raw))
    has_explanation = bool(_EXPLANATION_PATTERN.search(raw))

    # Guard: if "research" is used as a domain noun (e.g. "how is ML used in research?"),
    # do NOT count it as paper retrieval request!
    if has_paper_search and _RESEARCH_AS_DOMAIN.search(raw):
        # Check if there is an explicit request for papers (e.g. "papers", "publications", "articles")
        if not re.search(r"\b(papers?|publications?|articles?|literature|studies)\b", raw, re.IGNORECASE):
            has_paper_search = False

    paper_count = _extract_paper_count(raw) if has_paper_search else 5

    # 5. Resolve Intent
    if has_explanation and has_paper_search:
        # Both explanation and paper search present!
        topic = _extract_topic(raw, is_paper_search=True, is_mixed=True)
        return QueryAnalysis(
            raw_query=raw,
            intent=IntentType.MIXED_QUERY,
            topic=topic,
            paper_count=paper_count,
            is_comparison=is_comparison,
            is_applications=is_applications
        )

    if has_paper_search and not has_explanation:
        # Pure paper search query
        topic = _extract_topic(raw, is_paper_search=True, is_mixed=False)
        return QueryAnalysis(
            raw_query=raw,
            intent=IntentType.RESEARCH_PAPER_SEARCH,
            topic=topic,
            paper_count=paper_count,
            is_comparison=is_comparison,
            is_applications=is_applications
        )

    # 6. Check follow-ups (e.g. "Tell me more about it", "Give an example")
    is_follow_up = False
    topic = _extract_topic(raw)
    if history and re.search(r"\b(it|this|that|more|example|types?|detail)\b", raw, re.IGNORECASE) and len(raw.split()) <= 6:
        # Short query with pronouns referring to previous context
        for entry in reversed(history):
            prev_topic = entry.get("topic")
            if prev_topic:
                topic = prev_topic
                is_follow_up = True
                break

    # Default to GENERAL_KNOWLEDGE
    return QueryAnalysis(
        raw_query=raw,
        intent=IntentType.GENERAL_KNOWLEDGE,
        topic=topic,
        paper_count=5,
        is_comparison=is_comparison,
        is_applications=is_applications,
        is_follow_up=is_follow_up,
        concept_a=concept_a,
        concept_b=concept_b
    )
