// Prediction Form Event Listener
document.getElementById("predict-form")?.addEventListener("submit", async function (e) {
    e.preventDefault();

    const ticker = document.getElementById("ticker").value;
    const startDate = document.getElementById("start_date").value;
    const endDate = document.getElementById("end_date").value;

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, start_date: startDate, end_date: endDate })
    });

    const data = await response.json();
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = '';

    if (response.ok) {
        data.forEach(item => {
            resultsDiv.innerHTML += `<p>Date: ${item.date}, Predicted Price: ${item.predicted_price}</p>`;
        });
    } else {
        resultsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
    }
});

// Chatbox Form Event Listener
document.getElementById("chat-form")?.addEventListener("submit", function (e) {
    e.preventDefault();
    const userInputElement = document.getElementById("user_input");
    const userInput = userInputElement?.value.trim();

    if (!userInput) {
        alert("Prompt cannot be empty!");
        return;
    }

    appendMessage("User", userInput); // Display user's message in the chatbox
    userInputElement.value = ""; // Clear input field

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userInput }),
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error("Error in fetching response");
            }
            return response.json();
        })
        .then((data) => {
            if (data.error) {
                console.error("Chatbot error:", data.error);
                appendMessage("Error", data.error);
            } else {
                appendMessage("Model", data.response);
            }
        })
        .catch((err) => {
            console.error("Error communicating with chatbot:", err);
            appendMessage("Error", "Failed to communicate with chatbot.");
        });
});

function appendMessage(sender, message) {
    const chatBoxBody = document.getElementById("chatbox-body");
    if (!chatBoxBody) {
        console.error("Chatbox body element not found!");
        return;
    }
    const messageElement = document.createElement("div");
    messageElement.className = `message ${sender.toLowerCase()}`; // Add CSS class for styling
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatBoxBody.appendChild(messageElement);
    chatBoxBody.scrollTop = chatBoxBody.scrollHeight; // Scroll to the latest message
}
