/* =========================================
   RESEARCHBOT JAVASCRIPT CONTROLLER
   Scholara AI Research Assistant
========================================= */

let isProcessing = false;
let thinkingInterval = null;
let conversationHistory = [];

const thinkingMessages = [
    "Understanding your question...",
    "Analyzing research topics...",
    "Retrieving peer-reviewed papers...",
    "Synthesizing your answer..."
];


/* =========================================
   THINKING / LOADING INDICATOR
========================================= */

function showThinkingIndicator() {
    removeThinkingIndicator();

    const messages = document.getElementById("chatMessages");
    if (!messages) return;

    const thinkingDiv = document.createElement("div");
    thinkingDiv.className = "message bot-message thinking-message";
    thinkingDiv.id = "thinkingIndicator";

    thinkingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="bi bi-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-name">ResearchBot</div>
            <div class="thinking-content">
                <div class="thinking-status" id="thinkingText">
                    ${thinkingMessages[0]}
                </div>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;

    messages.appendChild(thinkingDiv);

    let messageIndex = 0;
    thinkingInterval = setInterval(() => {
        const thinkingText = document.getElementById("thinkingText");
        if (!thinkingText) return;

        messageIndex = (messageIndex + 1) % thinkingMessages.length;
        thinkingText.style.opacity = "0";

        setTimeout(() => {
            thinkingText.textContent = thinkingMessages[messageIndex];
            thinkingText.style.opacity = "1";
        }, 200);
    }, 2200);

    scrollToBottom();
}


function removeThinkingIndicator() {
    if (thinkingInterval) {
        clearInterval(thinkingInterval);
        thinkingInterval = null;
    }

    const thinkingIndicator = document.getElementById("thinkingIndicator");
    if (thinkingIndicator) {
        thinkingIndicator.remove();
    }
}


/* =========================================
   RENDER RESEARCH PAPER CARDS
========================================= */

function renderPaperCards(papers) {
    if (!papers || !Array.isArray(papers) || papers.length === 0) {
        return "";
    }

    let cardsHtml = '<div class="chatbot-paper-cards">';

    for (const paper of papers) {
        const safeTitle = escapeHtml(paper.title || "Untitled Paper");
        const safeAuthors = escapeHtml(paper.authors || "Unknown Authors");
        const safeJournal = paper.journal_conference ? escapeHtml(paper.journal_conference) : "";
        const safeAbstract = paper.abstract_snippet ? escapeHtml(paper.abstract_snippet) : "";
        const detailUrl = paper.detail_url || `/paper/${paper.id}`;

        const scoreBadge = (paper.score_percent !== undefined)
            ? `<span class="paper-score-badge"><i class="bi bi-graph-up"></i> ${paper.score_percent}% match</span>`
            : "";

        const yearBadge = paper.publication_year
            ? `<span class="paper-meta-badge"><i class="bi bi-calendar3"></i> ${paper.publication_year}</span>`
            : "";

        const journalBadge = safeJournal
            ? `<span class="paper-meta-badge"><i class="bi bi-journal-text"></i> ${safeJournal}</span>`
            : "";

        const doiBtn = paper.doi_url
            ? `<a href="${paper.doi_url}" target="_blank" rel="noopener" class="paper-action-btn outline"><i class="bi bi-link-45deg"></i> DOI</a>`
            : "";

        const pdfBtn = paper.pdf_link
            ? `<a href="${escapeHtml(paper.pdf_link)}" target="_blank" rel="noopener" class="paper-action-btn outline"><i class="bi bi-file-pdf"></i> PDF</a>`
            : "";

        const detailBtn = `<a href="${detailUrl}" target="_blank" class="paper-action-btn primary"><i class="bi bi-box-arrow-up-right"></i> View Details</a>`;

        const titleEscapedForJs = safeTitle.replace(/'/g, "\\'").replace(/"/g, '&quot;');
        const summarizeBtn = `<button type="button" class="paper-action-btn outline" onclick="summarizePaperById(${paper.id}, '${titleEscapedForJs}')"><i class="bi bi-card-text"></i> Summarize</button>`;

        cardsHtml += `
            <div class="chatbot-paper-card">
                <div class="paper-card-top">
                    <a href="${detailUrl}" target="_blank" class="paper-title-link">
                        ${safeTitle}
                    </a>
                    ${scoreBadge}
                </div>
                <div class="paper-authors">
                    <i class="bi bi-person"></i> ${safeAuthors}
                </div>
                <div class="paper-meta-row">
                    ${yearBadge}
                    ${journalBadge}
                </div>
                ${safeAbstract ? `<p class="paper-abstract">${safeAbstract}</p>` : ""}
                <div class="paper-actions-row">
                    ${detailBtn}
                    ${summarizeBtn}
                    ${doiBtn}
                    ${pdfBtn}
                </div>
            </div>
        `;
    }

    cardsHtml += "</div>";
    return cardsHtml;
}


/* Quick Summarize Action from Paper Card */
function summarizePaperById(paperId, title) {
    if (isProcessing) return;

    const input = document.getElementById("messageInput");
    if (!input) return;

    input.value = `Summarize the paper: "${title}" (ID: ${paperId})`;
    sendMessage();
}


/* =========================================
   SEND MESSAGE
========================================= */

async function sendMessage() {
    if (isProcessing) return;

    const input = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendButton");
    const message = input.value.trim();

    if (!message) return;

    isProcessing = true;
    if (sendButton) sendButton.disabled = true;

    // Display user message in UI
    addUserMessage(message);
    input.value = "";

    // Record user message in conversational context
    conversationHistory.push({
        role: "user",
        text: message
    });

    showThinkingIndicator();

    try {
        const response = await fetch("/api/chatbot", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                history: conversationHistory.slice(-6)
            })
        });

        const data = await response.json();
        removeThinkingIndicator();

        if (data.success) {
            addBotMessage(data.answer, data.papers, data.intent);

            // Record bot response in conversation history
            conversationHistory.push({
                role: "bot",
                text: data.answer,
                papers: data.papers || [],
                topic: data.topic,
                intent: data.intent
            });
        } else {
            addBotMessage("Sorry, an error occurred: " + (data.error || "Unknown error"));
        }
    } catch (error) {
        console.error("ResearchBot request error:", error);
        removeThinkingIndicator();
        addBotMessage("Sorry, I could not connect to ResearchBot. Please check your connection and try again.");
    } finally {
        isProcessing = false;
        if (sendButton) sendButton.disabled = false;
        input.focus();
    }
}


/* =========================================
   QUICK QUESTIONS
========================================= */

function askQuestion(question) {
    if (isProcessing) return;
    const input = document.getElementById("messageInput");
    if (!input) return;
    input.value = question;
    sendMessage();
}


/* =========================================
   USER MESSAGE RENDERING
========================================= */

function addUserMessage(message) {
    const messages = document.getElementById("chatMessages");
    if (!messages) return;

    const messageDiv = document.createElement("div");
    messageDiv.className = "message user-message";
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
        </div>
    `;

    messages.appendChild(messageDiv);
    scrollToBottom();
}


/* =========================================
   BOT MESSAGE RENDERING
========================================= */

function addBotMessage(message, papers = [], intent = "") {
    const messages = document.getElementById("chatMessages");
    if (!messages) return;

    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot-message";

    const paperCardsHtml = (papers && papers.length > 0)
        ? renderPaperCards(papers)
        : "";

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="bi bi-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-header-row">
                <div class="message-name">ResearchBot</div>
                <button class="speak-button" title="Listen to response" type="button">
                    <i class="bi bi-volume-up"></i>
                </button>
            </div>
            <div class="bot-response">
                <div class="bot-text-content">
                    ${formatResponse(message)}
                </div>
                ${paperCardsHtml}
            </div>
        </div>
    `;

    messages.appendChild(messageDiv);
    scrollToBottom();
}


/* =========================================
   MARKDOWN FORMATTING & HTML SANITIZING
========================================= */

function formatResponse(text) {
    if (!text) return "";
    if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
        const rawHtml = marked.parse(text, { breaks: true });
        return DOMPurify.sanitize(rawHtml);
    }
    return escapeHtml(text);
}


function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


/* =========================================
   SCROLL CHAT TO BOTTOM
========================================= */

function scrollToBottom() {
    const messages = document.getElementById("chatMessages");
    if (!messages) return;
    messages.scrollTop = messages.scrollHeight;
    window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
    });
}


/* =========================================
   ENTER KEY EVENT LISTENER
========================================= */

document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("messageInput");
    if (input) {
        input.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                sendMessage();
            }
        });
    }
});


/* =========================================
   FEATURE: VOICE INPUT (Speech-to-Text)
========================================= */

const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;
let voiceStatusHideTimer = null;

function getRecognition() {
    if (recognition || !SpeechRecognitionAPI) {
        return recognition;
    }

    recognition = new SpeechRecognitionAPI();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = function (event) {
        const transcript = event.results[0][0].transcript;
        const input = document.getElementById("messageInput");
        if (input) {
            input.value = transcript;
            input.focus();
        }
    };

    recognition.onerror = function () {
        setListeningState(false);
    };

    recognition.onend = function () {
        setListeningState(false);
    };

    return recognition;
}


function toggleVoiceInput() {
    if (!SpeechRecognitionAPI) {
        showVoiceStatus("Voice input is not supported in this browser.", true);
        return;
    }

    const rec = getRecognition();
    if (isListening) {
        rec.stop();
        return;
    }

    try {
        rec.start();
        setListeningState(true);
    } catch (error) {
        console.error("Speech recognition error:", error);
    }
}


function setListeningState(listening) {
    isListening = listening;
    const micButton = document.getElementById("micButton");
    if (micButton) {
        micButton.classList.toggle("mic-active", listening);
    }

    if (listening) {
        showVoiceStatus("Listening...", false);
    } else {
        hideVoiceStatus();
    }
}


function showVoiceStatus(text, isError) {
    const statusEl = document.getElementById("voiceStatus");
    const statusTextEl = document.getElementById("voiceStatusText");
    if (!statusEl || !statusTextEl) return;

    clearTimeout(voiceStatusHideTimer);
    statusTextEl.textContent = text;
    statusEl.classList.toggle("voice-status-error", !!isError);
    statusEl.style.display = "flex";

    if (isError) {
        voiceStatusHideTimer = setTimeout(hideVoiceStatus, 4000);
    }
}


function hideVoiceStatus() {
    const statusEl = document.getElementById("voiceStatus");
    if (!statusEl) return;
    statusEl.style.display = "none";
    statusEl.classList.remove("voice-status-error");
}


/* =========================================
   FEATURE: VOICE OUTPUT (Text-to-Speech)
========================================= */

let currentSpeakButton = null;

document.addEventListener("click", function (event) {
    const button = event.target.closest(".speak-button");
    if (!button) return;

    const messageContent = button.closest(".message-content");
    const textEl = messageContent
        ? (messageContent.querySelector(".bot-text-content") || messageContent.querySelector(".bot-response"))
        : null;

    const text = textEl ? textEl.textContent.trim() : "";
    toggleSpeak(text, button);
});


function toggleSpeak(text, button) {
    if (!("speechSynthesis" in window)) return;

    const wasSpeakingThisButton = currentSpeakButton === button && window.speechSynthesis.speaking;
    window.speechSynthesis.cancel();

    if (currentSpeakButton) {
        setSpeakButtonState(currentSpeakButton, false);
        currentSpeakButton = null;
    }

    if (wasSpeakingThisButton || !text) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;

    utterance.onend = function () {
        setSpeakButtonState(button, false);
        if (currentSpeakButton === button) currentSpeakButton = null;
    };

    utterance.onerror = function () {
        setSpeakButtonState(button, false);
        if (currentSpeakButton === button) currentSpeakButton = null;
    };

    window.speechSynthesis.speak(utterance);
    setSpeakButtonState(button, true);
    currentSpeakButton = button;
}


function setSpeakButtonState(button, speaking) {
    if (speaking) {
        button.classList.add("speaking");
        button.innerHTML = '<i class="bi bi-stop-fill"></i>';
        button.title = "Stop";
    } else {
        button.classList.remove("speaking");
        button.innerHTML = '<i class="bi bi-volume-up"></i>';
        button.title = "Listen to response";
    }
}
