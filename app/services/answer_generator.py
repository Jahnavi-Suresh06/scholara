"""
answer_generator.py

AI-Based Answer Generation and Knowledge Synthesis for ResearchBot.
Handles:
1. Gemini LLM generation when GEMINI_API_KEY is configured.
2. Robust, structured offline knowledge engine when Gemini is unavailable,
   handling general knowledge, comparisons, applications, and dynamic
   corpus-backed concept explanations from Scholara's 49k research papers.
3. Strict, factual paper summarization from paper metadata and abstracts.
"""

import os
import re
from typing import Dict, Any, Optional, List
from flask import current_app

from app import db
from app.models.paper import Paper
from app.services.query_classifier import IntentType, QueryAnalysis


# -----------------------------------------------------------------------------
# 1. CURATED KNOWLEDGE REPOSITORY FOR FUNDAMENTAL AI / CS CONCEPTS
# -----------------------------------------------------------------------------

_CONCEPT_KNOWLEDGE = {
    "machine learning": {
        "title": "Machine Learning (ML)",
        "definition": (
            "Machine Learning is a branch of Artificial Intelligence (AI) that enables computer systems "
            "to automatically learn and improve from experience and data without being explicitly programmed "
            "for every individual task. Instead of writing deterministic, rule-based code, algorithms optimize "
            "statistical parameters to recognize complex patterns and make data-driven predictions."
        ),
        "types": [
            "**Supervised Learning:** Algorithms train on labeled datasets containing input-output pairs to learn a mapping function (e.g., Linear Regression, Random Forests, Support Vector Machines).",
            "**Unsupervised Learning:** Algorithms uncover hidden patterns, intrinsic groupings, or latent representations from unlabeled data (e.g., K-Means Clustering, PCA, Autoencoders).",
            "**Reinforcement Learning:** Agents learn optimal policies through trial-and-error interactions with an environment, guided by scalar rewards and penalties (e.g., Q-Learning, PPO, Deep Q-Networks)."
        ],
        "workflow": [
            "1. **Data Collection & Preparation:** Gathering, cleaning, and preprocessing representative data.",
            "2. **Feature Engineering:** Selecting and extracting predictive attributes.",
            "3. **Model Selection & Training:** Optimizing model parameters using an objective/loss function.",
            "4. **Evaluation & Validation:** Measuring generalization on unseen test data using metrics such as Precision, Recall, F1-score, and ROC-AUC.",
            "5. **Inference & Deployment:** Serving the trained model in production systems."
        ],
        "applications": [
            "**Healthcare:** Disease diagnosis, patient risk stratification, and genomic sequencing analysis.",
            "**Finance:** Credit risk modeling, algorithmic trading, and real-time fraud detection.",
            "**Recommendation Systems:** Personalized content suggestions in streaming and e-commerce.",
            "**Autonomous Vehicles:** Sensor fusion, obstacle detection, and trajectory planning.",
            "**Natural Language Processing:** Machine translation, sentiment analysis, and search ranking."
        ]
    },
    "deep learning": {
        "title": "Deep Learning (DL)",
        "definition": (
            "Deep Learning is a specialized subfield of Machine Learning based on Artificial Neural Networks "
            "with multiple representation layers (hence 'deep'). By stacking non-linear processing layers, "
            "deep architectures automatically learn hierarchical feature representations directly from raw "
            "data, eliminating the need for manual feature engineering."
        ),
        "types": [
            "**Convolutional Neural Networks (CNNs):** Specialized for spatial grid data, utilizing convolutional kernels for computer vision and medical imaging.",
            "**Recurrent Neural Networks (RNNs, LSTMs, GRUs):** Tailored for temporal and sequential data with internal memory states for time-series and speech.",
            "**Transformers:** Utilize self-attention mechanisms to model long-range contextual dependencies across tokens in parallel, powering modern Large Language Models.",
            "**Generative Adversarial Networks (GANs) & Diffusion Models:** Generative architectures that synthesize realistic images, audio, and synthetic scientific data."
        ],
        "workflow": [
            "1. **Forward Propagation:** Raw input passes through successive hidden layers with non-linear activation functions.",
            "2. **Loss Calculation:** A loss function measures the discrepancy between model prediction and ground truth.",
            "3. **Backpropagation:** Gradients of the loss with respect to all layer weights are computed using the calculus chain rule.",
            "4. **Optimization:** Optimization algorithms (e.g., Adam, SGD with momentum) update weights to minimize loss."
        ],
        "applications": [
            "**Computer Vision:** Image segmentation, object detection, and facial recognition.",
            "**Speech Processing:** Automated speech recognition and expressive text-to-speech synthesis.",
            "**Drug Discovery & Bioinformatics:** Protein structure prediction (e.g., AlphaFold) and molecular property estimation.",
            "**Autonomous Robotics:** End-to-end vision-language-action policies."
        ]
    },
    "natural language processing": {
        "title": "Natural Language Processing (NLP)",
        "definition": (
            "Natural Language Processing (NLP) is an interdisciplinary field of Artificial Intelligence, Computer Science, "
            "and Computational Linguistics concerned with the interactions between computers and human language. "
            "Its ultimate objective is to enable computers to read, decipher, understand, and generate human languages "
            "in a valuable, context-aware manner."
        ),
        "types": [
            "**Natural Language Understanding (NLU):** Parsing grammar, semantic intent, named entities, and sentiment from input text.",
            "**Natural Language Generation (NLG):** Synthesizing coherent, grammatically correct, and contextually fluent text from structured representations."
        ],
        "workflow": [
            "1. **Text Preprocessing & Tokenization:** Subword tokenization (Byte-Pair Encoding, WordPiece) and normalization.",
            "2. **Vector Embeddings:** Mapping words/sentences into high-dimensional semantic vector spaces (e.g., Word2Vec, GloVe, SBERT).",
            "3. **Contextual Encoding:** Deep attention layers (Transformers) capture syntactic and semantic dependencies.",
            "4. **Task Head Execution:** Classification, sequence labeling, or autoregressive token generation."
        ],
        "applications": [
            "**Machine Translation:** Seamless multi-lingual translation systems.",
            "**Information Retrieval & Semantic Search:** Meaning-based document retrieval matching user queries against academic repositories.",
            "**Sentiment Analysis:** Opinion mining from reviews, social media, and customer support transcripts.",
            "**Conversational AI:** Intelligent research assistants, chatbots, and interactive agents."
        ]
    },
    "artificial intelligence": {
        "title": "Artificial Intelligence (AI)",
        "definition": (
            "Artificial Intelligence is a broad discipline within Computer Science focused on developing "
            "systems capable of performing tasks that typically require human cognition, including perception, "
            "reasoning, learning, problem-solving, and decision-making."
        ),
        "types": [
            "**Narrow / Weak AI (ANI):** AI specialized in a single predefined domain (e.g., chess engines, recommendation algorithms, facial recognition). All present-day AI belongs here.",
            "**General AI (AGI):** Hypothetical future AI capable of understanding, learning, and applying intelligence across diverse domains with human-level adaptability.",
            "**Superintelligent AI (ASI):** Hypothetical intelligence surpassing human cognitive capabilities across all disciplines."
        ],
        "workflow": [
            "AI encompasses multiple foundational subdisciplines:",
            "- **Machine Learning & Deep Learning:** Data-driven statistical pattern recognition.",
            "- **Knowledge Representation & Reasoning:** Ontologies, semantic graphs, and logic engines.",
            "- **Computer Vision:** Image and video perception and scene understanding.",
            "- **Robotics & Autonomous Control:** Physical environment sensing and actuation."
        ],
        "applications": [
            "**Healthcare & Medicine:** Precision medicine, automated radiology, and clinical decision support.",
            "**Scientific Discovery:** Accelerated material science simulations and climate modeling.",
            "**Finance & Commerce:** High-frequency risk assessment and market forecasting.",
            "**Everyday Technology:** Virtual assistants, predictive navigation, and smart automation."
        ]
    },
    "neural network": {
        "title": "Artificial Neural Networks (ANN)",
        "definition": (
            "An Artificial Neural Network is a computational framework inspired by the biological neural circuits of the "
            "human brain. It consists of interconnected processing units ('neurons' or nodes) organized into layers that "
            "transform input signals through weighted connections and non-linear activation functions."
        ),
        "types": [
            "**Feedforward Neural Networks (MLP):** Information flows strictly unidirectionally from input to output.",
            "**Convolutional Neural Networks (CNN):** Apply localized kernel filters to capture spatial spatial hierarchies.",
            "**Recurrent Neural Networks (RNN):** Contain feedback loops to maintain historical temporal context.",
            "**Transformer Networks:** Rely entirely on parallel attention mechanisms rather than recurrence."
        ],
        "workflow": [
            "1. **Input Layer:** Receives numerical feature vectors.",
            "2. **Linear Transformation:** Computes weighted sum: $z = \\sum (w_i \\cdot x_i) + b$.",
            "3. **Activation Function:** Introduces non-linearity (e.g., ReLU, GeLU, Sigmoid) so the network can learn non-linear decision boundaries.",
            "4. **Loss Calculation & Optimization:** Compares output to target and adjusts weights via gradient descent."
        ],
        "applications": [
            "Pattern recognition, speech synthesis, predictive modeling, and high-dimensional function approximation."
        ]
    },
    "cnn": {
        "title": "Convolutional Neural Networks (CNN)",
        "definition": (
            "A Convolutional Neural Network (CNN) is a deep neural network architecture designed specifically for processing "
            "data with a known grid-like structure, such as 2D images. CNNs leverage parameter sharing and spatial local receptive "
            "fields to achieve translation invariance and high computational efficiency."
        ),
        "types": [
            "**Classic Architectures:** LeNet-5, AlexNet, VGG.",
            "**Modern Architectures:** ResNet (Residual Connections), EfficientNet (Compound Scaling), DenseNet."
        ],
        "workflow": [
            "1. **Convolutional Layers:** Slide learnable mathematical filters (kernels) across input tensors to produce feature activation maps.",
            "2. **Non-Linearity (ReLU):** Applies element-wise activation to introduce non-linear expressiveness.",
            "3. **Pooling Layers (Max / Average):** Subsamples spatial dimensions, reducing parameters and conferring translation tolerance.",
            "4. **Fully Connected Layers:** Flattens high-level features for categorical classification or regression."
        ],
        "applications": [
            "Medical imaging (tumor detection, MRI analysis), autonomous vehicle vision, facial verification, and satellite imagery analysis."
        ]
    },
    "rnn": {
        "title": "Recurrent Neural Networks (RNN)",
        "definition": (
            "A Recurrent Neural Network (RNN) is a neural network designed for sequential data where the current output depends on both "
            "the current input and past information. Unlike feedforward networks, RNNs maintain an internal hidden state acting as a "
            "temporal memory across time steps."
        ),
        "types": [
            "**Standard Simple RNN:** Basic recurrent loop, prone to vanishing and exploding gradients over long horizons.",
            "**Long Short-Term Memory (LSTM):** Incorporates input, forget, and output gates with a cell state to preserve long-term dependencies.",
            "**Gated Recurrent Unit (GRU):** Streamlined variant combining cell state and hidden state using reset and update gates."
        ],
        "workflow": [
            "At each time step $t$, the hidden state $h_t$ is updated: $h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$.",
            "The updated hidden state is passed forward to time step $t+1$ and optionally emitted to an output layer."
        ],
        "applications": [
            "Time-series financial forecasting, automated speech recognition, sensor telemetry monitoring, and text generation."
        ]
    },
    "supervised learning": {
        "title": "Supervised Learning",
        "definition": (
            "Supervised Learning is the machine learning paradigm in which models are trained using labeled training datasets. "
            "Each training instance consists of an input vector $X$ and a corresponding ground-truth target label $y$. "
            "The objective is to approximate a mathematical mapping function $f(X) \\approx y$ that generalizes accurately to novel, unseen inputs."
        ),
        "types": [
            "**Classification:** Predicting discrete qualitative labels (e.g., spam vs. ham, tumor classification). Common algorithms: Logistic Regression, Random Forest, SVM, Deep Classifiers.",
            "**Regression:** Predicting continuous quantitative numerical values (e.g., house prices, stock indices). Common algorithms: Linear Regression, Ridge/Lasso, Gradient Boosted Trees (XGBoost)."
        ],
        "workflow": [
            "1. Dataset labeling and validation.",
            "2. Splitting into Train, Validation, and Test partitions.",
            "3. Minimizing empirical risk via loss functions (e.g., Cross-Entropy, Mean Squared Error).",
            "4. Hyperparameter tuning using Cross-Validation."
        ],
        "applications": [
            "Credit scoring, medical diagnostics, weather forecasting, and customer churn prediction."
        ]
    },
    "unsupervised learning": {
        "title": "Unsupervised Learning",
        "definition": (
            "Unsupervised Learning is the paradigm where algorithms are provided with data that has no associated target labels. "
            "The model must independently discern the underlying structure, clusters, probability densities, or low-dimensional "
            "manifolds embedded within the input data."
        ),
        "types": [
            "**Clustering:** Grouping similar data points based on geometric distance metrics (e.g., K-Means, DBSCAN, Hierarchical Clustering).",
            "**Dimensionality Reduction:** Compressing high-dimensional data while preserving maximum variance or local geometric topology (e.g., PCA, t-SNE, UMAP).",
            "**Anomaly Detection:** Identifying outliers that deviate significantly from standard distribution baselines (e.g., Isolation Forest, One-Class SVM)."
        ],
        "workflow": [
            "1. Feature scaling and distance normalization.",
            "2. Optimizing unsupervised criteria (e.g., cluster inertia, reconstruction error, or variance retention).",
            "3. Interpreting qualitative latent clusters and embeddings."
        ],
        "applications": [
            "Customer market segmentation, gene expression clustering, network intrusion detection, and data compression."
        ]
    },
    "sentiment analysis": {
        "title": "Sentiment Analysis",
        "definition": (
            "Sentiment Analysis (also known as opinion mining) is an NLP technique used to systematically identify, extract, "
            "quantify, and study affective states and subjective information from text resources. It determines whether emotional "
            "tone is positive, negative, neutral, or fine-grained (e.g., satisfaction, frustration)."
        ),
        "types": [
            "**Rule/Lexicon-Based:** Uses predetermined sentiment dictionaries (e.g., VADER, SentiWordNet).",
            "**Supervised Classification:** Uses Naive Bayes, SVMs, or Logistic Regression on TF-IDF features.",
            "**Deep Contextual Models:** Uses fine-tuned Transformer models (BERT, RoBERTa) capturing subtle sarcasm and negation."
        ],
        "workflow": [
            "1. Text tokenization and cleaning.",
            "2. Handling linguistic negation and intensifiers.",
            "3. Computing contextual word embeddings.",
            "4. Classification into discrete sentiment polarities."
        ],
        "applications": [
            "Brand reputation management, customer feedback analysis, financial market sentiment tracking, and public health surveys."
        ]
    }
}


# -----------------------------------------------------------------------------
# 2. LOCAL KNOWLEDGE FORMATTERS
# -----------------------------------------------------------------------------

def _format_concept_explanation(concept_info: dict) -> str:
    """Format a concept entry into a polished markdown academic response."""
    lines = [
        f"### {concept_info['title']}",
        "",
        concept_info["definition"],
        "",
        "#### Key Categories & Subtypes",
    ]
    for item in concept_info.get("types", []):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("#### How It Works")
    for step in concept_info.get("workflow", []):
        lines.append(f"{step}")

    lines.append("")
    lines.append("#### Real-World Applications")
    for app in concept_info.get("applications", []):
        lines.append(f"- {app}")

    return "\n".join(lines)


def _generate_comparison_answer(concept_a: str, concept_b: str) -> str:
    """Generate a structured comparative analysis between two concepts."""
    a_lower = (concept_a or "").lower().strip()
    b_lower = (concept_b or "").lower().strip()

    # Special case: AI vs ML
    if ("ai" in a_lower or "artificial intelligence" in a_lower) and ("ml" in b_lower or "machine learning" in b_lower) or \
       ("ai" in b_lower or "artificial intelligence" in b_lower) and ("ml" in a_lower or "machine learning" in a_lower):
        return """### Comparison: Artificial Intelligence (AI) vs. Machine Learning (ML)

While often used interchangeably in casual discourse, **Artificial Intelligence** and **Machine Learning** represent different levels of abstraction within computer science.

#### Core Definitions
- **Artificial Intelligence (AI):** The overarching scientific discipline and vision of creating systems capable of simulating human intelligence, reasoning, decision-making, and problem-solving.
- **Machine Learning (ML):** A specific, foundational subset of AI that focuses on enabling systems to learn patterns and optimize performance directly from data without deterministic, hardcoded rules.

#### Key Differences

| Dimension | Artificial Intelligence (AI) | Machine Learning (ML) |
|---|---|---|
| **Scope** | Broad umbrella field covering reasoning, vision, robotics, NLP, and logic. | A specialized subfield focused exclusively on data-driven learning. |
| **Primary Goal** | Simulate human-like intelligence, decision-making, and rational behavior. | Learn predictive mathematical models from historical and empirical data. |
| **Approach** | Includes rule-based expert systems, heuristic search, logic, and ML. | Relies strictly on statistical learning, loss optimization, and neural nets. |
| **Data Requirement** | Can operate using symbolic logic and rule engines without big data. | Fundamentally dependent on representative training datasets. |
| **Adaptability** | Can be deterministic (e.g. A* pathfinding) or adaptive. | Continuously refines parameters when supplied with new training data. |

#### Illustrative Relationship
Think of AI as the broad universe of intelligent automation. Machine Learning is the statistical engine driving modern AI breakthroughs. Deep Learning is an even deeper subset of Machine Learning."""

    # Special case: CNN vs RNN
    if ("cnn" in a_lower or "convolutional" in a_lower) and ("rnn" in b_lower or "recurrent" in b_lower) or \
       ("cnn" in b_lower or "convolutional" in b_lower) and ("rnn" in a_lower or "recurrent" in a_lower):
        return """### Comparison: Convolutional Neural Networks (CNN) vs. Recurrent Neural Networks (RNN)

CNNs and RNNs are two specialized deep learning architectures designed for fundamentally different types of structured data.

#### Core Distinction
- **CNNs** specialize in **spatial data** with local geometric hierarchies (e.g., 2D image matrices, spectrograms).
- **RNNs** specialize in **sequential and temporal data** where order and history matter (e.g., text, speech, time-series).

#### Key Differences

| Architectural Feature | Convolutional Neural Network (CNN) | Recurrent Neural Network (RNN) |
|---|---|---|
| **Primary Data Type** | Spatial grid data (images, video frames). | Sequential data (natural language, time series, audio). |
| **Core Mechanism** | Convolutional filters/kernels sliding across receptive fields. | Recurrent loops maintaining an internal hidden state memory across steps. |
| **Parameter Sharing** | Spatial parameter sharing (same filter applied across whole image). | Temporal parameter sharing (same weight matrices applied across all time steps). |
| **Computation** | Highly parallelizable across tensor dimensions on GPUs. | Inherently sequential (step $t$ requires hidden state from step $t-1$). |
| **Key Advantage** | Translation invariance and hierarchical spatial feature extraction. | Dynamic sequence length handling and temporal context modeling. |
| **Main Limitation** | Cannot naturally model sequential recurrence or variable-length history. | Susceptible to vanishing/exploding gradients across long time horizons. |

#### When to Choose Which?
- Choose **CNN** when analyzing spatial structures (e.g., medical image diagnosis, satellite classification, face recognition).
- Choose **RNN** (or LSTM/GRU) when modeling order-dependent sequences (e.g., stock trend prediction, sensor telemetry, speech recognition)."""

    # Supervised vs Unsupervised
    if ("supervised" in a_lower and "unsupervised" in b_lower) or ("supervised" in b_lower and "unsupervised" in a_lower):
        return """### Comparison: Supervised Learning vs. Unsupervised Learning

Supervised and Unsupervised Learning represent the two foundational paradigms of machine learning, separated primarily by the presence of ground-truth target labels.

#### Key Differences

| Aspect | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Training Data** | Labeled data: Input-target pairs $(X, y)$. | Unlabeled data: Input features $X$ only. |
| **Objective** | Learn a mapping function $f(X) \\approx y$ to predict labels on unseen data. | Discover latent structures, groupings, or probability distributions in $X$. |
| **Feedback Mechanism** | Explicit error feedback via loss computation against ground truth. | No external error signal; evaluates internal cohesion or variance retention. |
| **Core Tasks** | Classification (discrete labels) & Regression (continuous values). | Clustering, Dimensionality Reduction, Anomaly Detection. |
| **Common Algorithms** | Random Forests, SVMs, Logistic Regression, XGBoost, Deep Neural Nets. | K-Means, DBSCAN, PCA, Autoencoders, Isolation Forests. |
| **Human Effort** | High (requires manual annotation/labeling of training samples). | Low (operates directly on raw, unannotated datasets). |

#### Practical Summary
- Use **Supervised Learning** when you have historical examples of the exact target outcome you wish to predict.
- Use **Unsupervised Learning** when exploring unstructured data to uncover customer segments, detect novel fraud anomalies, or compress high-dimensional features."""

    # Generic comparison fallback
    return f"""### Conceptual Comparison: {concept_a.title()} vs. {concept_b.title()}

#### Overview
Both **{concept_a.title()}** and **{concept_b.title()}** represent significant technical paradigms in modern computational science.

- **{concept_a.title()}:** Focuses on distinct principles and workflows tailored to its specific operational requirements and problem space.
- **{concept_b.title()}:** Addresses complementary computational objectives, employing different algorithmic structures and assumptions.

#### Analytical Comparison
1. **Target Problem Space:** Each approach is optimized for distinct modalities and data structures.
2. **Computational Architecture:** Algorithms and training procedures vary significantly in complexity, optimization objectives, and data dependencies.
3. **Practical Deployment:** In modern engineering pipelines, both techniques are frequently combined in hybrid architectures to maximize end-to-end system efficacy."""


def _generate_applications_answer(topic: str) -> str:
    """Generate a structured breakdown of real-world applications for a given technical domain."""
    topic_clean = topic.strip().title()
    return f"""### Real-World Applications of {topic_clean}

{topic_clean} plays a transformative role across contemporary industry, scientific inquiry, and technological infrastructure:

#### 1. Healthcare & Biomedical Sciences
- **Automated Diagnostic Imaging:** Enhancing early detection of pathologies in CT, MRI, and histological scans.
- **Precision Medicine & Genomics:** Identifying genetic markers and tailoring personalized treatment protocols.
- **Drug Discovery & Molecular Design:** Accelerating candidate drug screening and bio-molecular simulation.

#### 2. Autonomous Systems & Robotics
- **Self-Driving Vehicles:** Real-time sensor fusion, path optimization, and collision avoidance.
- **Industrial Automation:** Robotic manipulators, predictive maintenance, and quality inspection on manufacturing lines.

#### 3. Finance & Economic Intelligence
- **Algorithmic Fraud Detection:** Monitoring transactional streams in real time to intercept unauthorized activities.
- **Automated Risk Stratification:** Assessing credit risk, insurance claims, and market volatility modeling.

#### 4. Natural Language & Cognitive Interfaces
- **Virtual Research Assistants:** Synthesizing academic literature, answering questions, and summarizing complex documents.
- **Real-Time Cross-Lingual Translation:** Facilitating global collaboration and multi-lingual access to scientific data.

#### 5. Scientific Discovery & Climate Science
- **Climate & Weather Forecasting:** Processing high-dimensional satellite datasets for extreme weather prediction.
- **Materials Science:** Accelerating the discovery of high-performance battery electrolytes and semiconductor compounds."""


# -----------------------------------------------------------------------------
# 3. DYNAMIC CORPUS SYNTHESIS (FOR ANY UNSEEN ACADEMIC TOPIC)
# -----------------------------------------------------------------------------

def _synthesize_from_corpus(topic: str) -> Optional[str]:
    """
    Dynamically synthesizes an academic overview of any topic by inspecting
    peer-reviewed papers in Scholara's 49k paper database.
    Ensures that ResearchBot can explain ANY technical concept dynamically
    without hardcoding!
    """
    try:
        from app.services.paper_retrieval_service import retrieve_papers_for_query
        retrieval = retrieve_papers_for_query(topic, top_k=3, min_score=0.35)
        papers = retrieval.get("papers", [])

        if not papers:
            return None

        # Extract relevant academic context from real abstracts
        key_insights = []
        for p in papers:
            abstract = (p.get("abstract") or "").strip()
            # Extract first two sentences of the abstract
            sentences = re.split(r"(?<=[.!?])\s+", abstract)
            first_sentences = " ".join(sentences[:2]) if sentences else abstract[:180]
            key_insights.append(f"- **{p['title']}** ({p.get('publication_year', 'Recent')}): {first_sentences}")

        insights_text = "\n".join(key_insights)
        topic_title = topic.strip().title()

        return f"""### Overview: {topic_title}

**{topic_title}** is an active area of investigation within computer science and empirical research.

#### Core Concept & Research Focus
In modern scientific literature, work in {topic_title} addresses challenging computational, theoretical, and applied engineering problems. Researchers focus on formulating robust models, optimizing algorithmic performance, and scaling implementations for real-world complexity.

#### Key Themes from the Scientific Literature
{insights_text}

#### Methodological Foundations
- **Mathematical Formulations:** Rigorous theoretical foundations guiding model convergence and stability.
- **Empirical Validation:** Benchmarking against standardized datasets using rigorous quantitative metrics.
- **Scalability & Efficiency:** Designing low-latency, computationally feasible architectures for deployment."""
    except Exception as e:
        current_app.logger.warning(f"Corpus synthesis fallback error: {e}")
        return None


# -----------------------------------------------------------------------------
# 4. PAPER SUMMARIZATION (STRICT & FACTUAL)
# -----------------------------------------------------------------------------

def summarize_paper(paper: Paper) -> str:
    """
    Summarizes a paper strictly from its metadata and abstract.
    Never invents or hallucinates facts.
    """
    if not paper:
        return "Paper details not found in the database."

    title = paper.title or "Untitled Research Paper"
    authors = paper.authors or "Unknown Authors"
    year = f" ({paper.publication_year})" if paper.publication_year else ""
    pub = f"Published in: *{paper.journal_conference}*" if paper.journal_conference else ""
    doi_info = f"DOI: `{paper.doi}`" if paper.doi else ""
    abstract = (paper.abstract or "").strip()

    if not abstract:
        return f"""### Research Paper Summary: {title}
- **Authors:** {authors}{year}
- {pub}
- {doi_info}

*Note: No abstract text was available in the database for this publication.*"""

    # Segment abstract sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) > 10]

    # Structure into academic components
    objective = sentences[0] if sentences else abstract
    methodology = ""
    findings = ""

    if len(sentences) >= 3:
        # Middle sentences typically discuss methodology
        methodology = " ".join(sentences[1:-1])
        # Last sentence typically reports findings / conclusions
        findings = sentences[-1]
    elif len(sentences) == 2:
        methodology = sentences[0]
        findings = sentences[1]
    else:
        methodology = abstract
        findings = "Detailed in the full publication."

    lines = [
        f"### Research Paper Summary",
        f"**{title}**",
        "",
        f"- **Authors:** {authors}{year}",
    ]
    if pub:
        lines.append(f"- {pub}")
    if doi_info:
        lines.append(f"- {doi_info}")

    lines.extend([
        "",
        "#### 1. Research Objective",
        objective,
        "",
        "#### 2. Proposed Methodology & Approach",
        methodology,
        "",
        "#### 3. Key Findings & Contributions",
        findings,
        "",
        f"[View Full Paper Details in Scholara](/paper/{paper.id})"
    ])

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 5. GEMINI API CLIENT (LAZY & SAFE)
# -----------------------------------------------------------------------------

def _call_gemini_api(prompt: str, system_instruction: str = "") -> str:
    """
    Calls Gemini API using google-genai SDK if GEMINI_API_KEY is configured.
    Falls back gracefully if the API fails, rate-limits, or has no key.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    from google import genai
    client = genai.Client(api_key=api_key)

    # Models in order of preference
    model_candidates = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "gemini-1.5-flash",
        "gemini-2.0-flash",
    ]

    last_error = None
    for model_name in model_candidates:
        try:
            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            current_app.logger.warning(f"Gemini model {model_name} failed: {e}")

    raise RuntimeError(f"All Gemini models failed: {last_error}")


# -----------------------------------------------------------------------------
# 6. UNIFIED ANSWER GENERATION DISPATCHER
# -----------------------------------------------------------------------------

def generate_answer(
    analysis: QueryAnalysis,
    history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a high-quality explanation based on user query analysis.
    Tries Gemini API first when configured; falls back gracefully to
    local knowledge / dynamic corpus synthesis without breaking or erroring.
    """
    if analysis.is_empty:
        return "Please enter a question or research topic to get started."

    raw_query = analysis.raw_query
    topic = analysis.topic or raw_query
    topic_key = topic.lower().strip()

    # Special Case: PAPER_SUMMARIZATION
    if analysis.intent == IntentType.PAPER_SUMMARIZATION:
        paper = None
        if analysis.paper_id:
            paper = db.session.get(Paper, analysis.paper_id)
        elif analysis.paper_title:
            paper = Paper.query.filter(Paper.title.ilike(f"%{analysis.paper_title}%")).first()

        if paper:
            # Try Gemini first for paper summarization if available
            try:
                system_inst = (
                    "You are ResearchBot. Summarize this research paper accurately and factually. "
                    "Do NOT invent any details not present in the title and abstract."
                )
                prompt = (
                    f"Paper Title: {paper.title}\n"
                    f"Authors: {paper.authors}\n"
                    f"Year: {paper.publication_year}\n"
                    f"Abstract: {paper.abstract}\n\n"
                    "Provide a structured summary with headings for Research Objective, Methodology, Key Findings, and Research Significance."
                )
                return _call_gemini_api(prompt, system_instruction=system_inst)
            except Exception:
                return summarize_paper(paper)
        else:
            return (
                "Please specify which paper you would like me to summarize (e.g., provide a paper title or ID), "
                "or search for papers first (e.g., *'Find research papers on Machine Learning'*), "
                "and then click the **Summarize** button on any paper card!"
            )

    # 1. Try Gemini API first for General Knowledge or Mixed Explanations
    try:
        system_inst = """You are ResearchBot, an AI research assistant for Scholara.
Rules:
- Give a direct, accurate, structured explanation with headings and bullet points.
- Explain concepts in simple, clear language with intuitive examples.
- For comparisons, highlight key differences in a structured format.
- For application questions, list concrete real-world use cases.
- Do NOT list or invent research paper recommendations in your text (the system provides paper cards separately when appropriate)."""

        history_context = ""
        if history:
            prev_turns = []
            for h in history[-3:]:
                role = "User" if h.get("role") == "user" else "Assistant"
                prev_turns.append(f"{role}: {h.get('text', '')}")
            history_context = "Conversation history:\n" + "\n".join(prev_turns) + "\n\n"

        prompt = f"{history_context}User Query: {raw_query}"
        gemini_text = _call_gemini_api(prompt, system_instruction=system_inst)
        if gemini_text:
            return gemini_text
    except Exception as e:
        current_app.logger.info(f"Using local knowledge engine (Gemini unavailable: {e})")

    # 2. Local Knowledge Engine Fallback

    # Comparison
    if analysis.is_comparison and (analysis.concept_a or " vs " in raw_query.lower() or "difference between" in raw_query.lower()):
        cA = analysis.concept_a or "Concept A"
        cB = analysis.concept_b or "Concept B"
        return _generate_comparison_answer(cA, cB)

    # Applications
    if analysis.is_applications and ("application" in raw_query.lower() or "use cases" in raw_query.lower()):
        return _generate_applications_answer(topic)

    # Curated knowledge lookup
    for key, info in _CONCEPT_KNOWLEDGE.items():
        if key == topic_key or key in topic_key or topic_key in key or (topic_key == "nlp" and "natural language processing" in key):
            return _format_concept_explanation(info)

    # Dynamic Corpus Synthesis from Scholara's 49k Papers
    corpus_summary = _synthesize_from_corpus(topic)
    if corpus_summary:
        return corpus_summary

    # General fallback explanation
    return f"""### {topic.strip().title()}

**{topic.strip().title()}** is an important concept in modern computing and academic research.

#### Overview
This area focuses on understanding underlying computational principles, developing optimized algorithmic workflows, and addressing practical challenges in real-world implementations.

#### Key Principles
- **Theoretical Foundation:** Grounded in mathematical models, statistical inference, and computational theory.
- **Implementation:** Leverages scalable software architectures and specialized hardware accelerators.
- **Continuous Innovation:** Rapidly evolving through active peer-reviewed research and open-source benchmarks.

You can also ask me for specific research papers on this topic by asking:
> *"Find research papers on {topic.strip()}"*"""
