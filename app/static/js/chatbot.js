/* =========================================
   RESEARCHBOT THINKING / LOADING STATE
========================================= */

let isProcessing = false;
let thinkingInterval = null;

const thinkingMessages = [
    "Searching relevant research papers...",
    "Analyzing research context...",
    "Generating your answer..."
];


function showThinkingIndicator() {

    // Prevent duplicate thinking indicators
    removeThinkingIndicator();

    const messages = document.getElementById("chatMessages");

    if (!messages) {
        return;
    }

    const thinkingDiv = document.createElement("div");

    thinkingDiv.className = "message bot-message thinking-message";
    thinkingDiv.id = "thinkingIndicator";

    thinkingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="bi bi-robot"></i>
        </div>

        <div class="message-content">

            <div class="message-name">
                ResearchBot
            </div>

            <div class="thinking-content">

                <div
                    class="thinking-status"
                    id="thinkingText"
                >
                    Searching relevant research papers...
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

        const thinkingText =
            document.getElementById("thinkingText");

        if (!thinkingText) {
            return;
        }

        messageIndex =
            (messageIndex + 1) % thinkingMessages.length;

        thinkingText.style.opacity = "0";

        setTimeout(() => {

            thinkingText.textContent =
                thinkingMessages[messageIndex];

            thinkingText.style.opacity = "1";

        }, 200);

    }, 2500);

    scrollToBottom();
}


function removeThinkingIndicator() {

    if (thinkingInterval) {

        clearInterval(thinkingInterval);

        thinkingInterval = null;

    }

    const thinkingIndicator =
        document.getElementById("thinkingIndicator");

    if (thinkingIndicator) {

        thinkingIndicator.remove();

    }
}


/* =========================================
   SEND MESSAGE
========================================= */

async function sendMessage() {

    // Prevent multiple requests while ResearchBot is processing
    if (isProcessing) {
        return;
    }

    const input =
        document.getElementById("messageInput");

    const sendButton =
        document.getElementById("sendButton");

    const message = input.value.trim();

    if (!message) {
        return;
    }


    // Lock sending while processing

    isProcessing = true;

    if (sendButton) {
        sendButton.disabled = true;
    }


    // Display user message

    addUserMessage(message);

    input.value = "";


    // Show thinking indicator immediately

    showThinkingIndicator();


    try {

        const response = await fetch("/api/chatbot", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message
            })

        });


        const data = await response.json();


        // Remove thinking indicator before displaying response

        removeThinkingIndicator();


        if (data.success) {

            addBotMessage(data.answer);

        } else {

            addBotMessage(
                "Sorry, something went wrong: " +
                (data.error || "Unknown error")
            );

        }

    } catch (error) {

        console.error(error);


        // Remove thinking indicator if connection fails

        removeThinkingIndicator();


        addBotMessage(
            "Sorry, I couldn't connect to ResearchBot. Please try again."
        );

    } finally {

        // Allow sending again

        isProcessing = false;

        if (sendButton) {
            sendButton.disabled = false;
        }

        input.focus();

    }
}


/* =========================================
   QUICK QUESTIONS
========================================= */

function askQuestion(question) {

    // Prevent quick actions during processing
    if (isProcessing) {
        return;
    }

    const input =
        document.getElementById("messageInput");

    input.value = question;

    sendMessage();
}


/* =========================================
   USER MESSAGE
========================================= */

function addUserMessage(message) {

    const messages =
        document.getElementById("chatMessages");

    const messageDiv =
        document.createElement("div");

    messageDiv.className =
        "message user-message";

    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
        </div>
    `;

    messages.appendChild(messageDiv);

    scrollToBottom();
}


/* =========================================
   RESEARCHBOT MESSAGE
========================================= */

function addBotMessage(message) {

    const messages =
        document.getElementById("chatMessages");

    const messageDiv =
        document.createElement("div");

    messageDiv.className =
        "message bot-message";

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="bi bi-robot"></i>
        </div>

        <div class="message-content">

            <div class="message-header-row">

                <div class="message-name">
                    ResearchBot
                </div>

                <button
                    class="speak-button"
                    title="Listen to response"
                    type="button"
                >
                    <i class="bi bi-volume-up"></i>
                </button>

            </div>

            <div class="bot-response">
                ${formatResponse(message)}
            </div>

        </div>
    `;

    messages.appendChild(messageDiv);

    scrollToBottom();
}


/* =========================================
   MARKDOWN FORMATTING
========================================= */

function formatResponse(text) {

    // Parse Markdown safely

    const rawHtml =
        marked.parse(text, {
            breaks: true
        });

    // Sanitize generated HTML for XSS safety

    return DOMPurify.sanitize(rawHtml);
}


/* =========================================
   HTML ESCAPING
========================================= */

function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}


/* =========================================
   SCROLL CHAT TO BOTTOM
========================================= */

function scrollToBottom() {

    const messages =
        document.getElementById("chatMessages");

    if (!messages) {
        return;
    }

    messages.scrollTop =
        messages.scrollHeight;

    window.scrollTo({

        top: document.body.scrollHeight,

        behavior: "smooth"

    });
}


/* =========================================
   PRESS ENTER TO SEND
========================================= */

document.addEventListener(
    "DOMContentLoaded",
    function () {

        const input =
            document.getElementById("messageInput");

        if (input) {

            input.addEventListener(
                "keydown",
                function (event) {

                    if (event.key === "Enter") {

                        event.preventDefault();

                        sendMessage();

                    }

                }
            );

        }

    }
);


/* =========================================
   FEATURE: VOICE INPUT
   Speech-to-Text
========================================= */

const SpeechRecognitionAPI =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;


let recognition = null;

let isListening = false;

let voiceStatusHideTimer = null;


function getRecognition() {

    if (
        recognition ||
        !SpeechRecognitionAPI
    ) {

        return recognition;

    }


    recognition =
        new SpeechRecognitionAPI();


    recognition.continuous = false;

    recognition.interimResults = false;

    recognition.lang = "en-US";


    recognition.onresult =
        function (event) {

            const transcript =
                event.results[0][0].transcript;


            const input =
                document.getElementById(
                    "messageInput"
                );


            if (input) {

                input.value = transcript;

                input.focus();

            }

        };


    recognition.onerror =
        function () {

            setListeningState(false);

        };


    recognition.onend =
        function () {

            setListeningState(false);

        };


    return recognition;

}


/* Toggle microphone */

function toggleVoiceInput() {

    if (!SpeechRecognitionAPI) {

        showVoiceStatus(
            "Voice input is not supported in this browser.",
            true
        );

        return;

    }


    const rec =
        getRecognition();


    if (isListening) {

        rec.stop();

        return;

    }


    try {

        rec.start();

        setListeningState(true);

    } catch (error) {

        console.error(error);

    }

}


/* Update microphone state */

function setListeningState(listening) {

    isListening = listening;


    const micButton =
        document.getElementById("micButton");


    if (micButton) {

        micButton.classList.toggle(
            "mic-active",
            listening
        );

    }


    if (listening) {

        showVoiceStatus(
            "Listening...",
            false
        );

    } else {

        hideVoiceStatus();

    }

}


/* Show voice status */

function showVoiceStatus(
    text,
    isError
) {

    const statusEl =
        document.getElementById("voiceStatus");


    const statusTextEl =
        document.getElementById(
            "voiceStatusText"
        );


    if (
        !statusEl ||
        !statusTextEl
    ) {

        return;

    }


    clearTimeout(
        voiceStatusHideTimer
    );


    statusTextEl.textContent =
        text;


    statusEl.classList.toggle(
        "voice-status-error",
        !!isError
    );


    statusEl.style.display =
        "flex";


    if (isError) {

        voiceStatusHideTimer =
            setTimeout(
                hideVoiceStatus,
                4000
            );

    }

}


/* Hide voice status */

function hideVoiceStatus() {

    const statusEl =
        document.getElementById(
            "voiceStatus"
        );


    if (!statusEl) {

        return;

    }


    statusEl.style.display =
        "none";


    statusEl.classList.remove(
        "voice-status-error"
    );

}


/* =========================================
   FEATURE: VOICE OUTPUT
   Text-to-Speech
========================================= */

let currentSpeakButton = null;


document.addEventListener(
    "click",
    function (event) {

        const button =
            event.target.closest(
                ".speak-button"
            );


        if (!button) {

            return;

        }


        const messageContent =
            button.closest(
                ".message-content"
            );


        const responseEl =
            messageContent
                ? messageContent.querySelector(
                    ".bot-response"
                )
                : null;


        const text =
            responseEl
                ? responseEl.textContent.trim()
                : "";


        toggleSpeak(
            text,
            button
        );

    }
);


/* Start or stop speech */

function toggleSpeak(
    text,
    button
) {

    if (
        !(
            "speechSynthesis"
            in window
        )
    ) {

        return;

    }


    const wasSpeakingThisButton =

        currentSpeakButton === button &&

        window.speechSynthesis.speaking;


    // Stop any existing speech

    window.speechSynthesis.cancel();


    if (currentSpeakButton) {

        setSpeakButtonState(
            currentSpeakButton,
            false
        );


        currentSpeakButton = null;

    }


    // Stop if clicking the currently speaking message

    if (
        wasSpeakingThisButton ||
        !text
    ) {

        return;

    }


    const utterance =
        new SpeechSynthesisUtterance(
            text
        );


    utterance.rate = 1;


    utterance.onend =
        function () {

            setSpeakButtonState(
                button,
                false
            );


            if (
                currentSpeakButton === button
            ) {

                currentSpeakButton = null;

            }

        };


    utterance.onerror =
        function () {

            setSpeakButtonState(
                button,
                false
            );


            if (
                currentSpeakButton === button
            ) {

                currentSpeakButton = null;

            }

        };


    window.speechSynthesis.speak(
        utterance
    );


    setSpeakButtonState(
        button,
        true
    );


    currentSpeakButton = button;

}


/* Update speaker button */

function setSpeakButtonState(
    button,
    speaking
) {

    if (speaking) {

        button.classList.add(
            "speaking"
        );


        button.innerHTML =
            '<i class="bi bi-stop-fill"></i>';


        button.title = "Stop";

    } else {

        button.classList.remove(
            "speaking"
        );


        button.innerHTML =
            '<i class="bi bi-volume-up"></i>';


        button.title =
            "Listen to response";

    }

}