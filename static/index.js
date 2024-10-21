// Initial greeting when the page loads
window.onload = function() {
    var chatBox = document.getElementById("chat-box");
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/start_conversation", true);
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var botMessage = document.createElement("div");
            botMessage.textContent = "Bot: " + JSON.parse(xhr.responseText).response;
            botMessage.classList.add("bot-message");
            chatBox.appendChild(botMessage);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    };
    xhr.send();
};

// Function to handle sending a message
function sendMessage() {
    var userInput = document.getElementById("user-input").value.trim();
    if (!userInput) return;

    var chatBox = document.getElementById("chat-box");

    // Add the user's message to the chat
    var userMessage = document.createElement("div");
    userMessage.textContent = "You: " + userInput;
    userMessage.classList.add("user-message");
    chatBox.appendChild(userMessage);

    // Clear the input field
    document.getElementById("user-input").value = "";

    // Show the typing indicator
    var typingIndicator = document.getElementById("typing-indicator");
    typingIndicator.style.display = "block";

    // Send the message to the server and get the bot's response
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/get_response", true);
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var botResponse = JSON.parse(xhr.responseText);
            var botMessage = document.createElement("div");
            botMessage.textContent = "Bot: " + botResponse.response;
            botMessage.classList.add("bot-message");
            chatBox.appendChild(botMessage);
            chatBox.scrollTop = chatBox.scrollHeight;

            // Hide the typing indicator
            typingIndicator.style.display = "none";

            // Disable input if the bot says goodbye (conversation ended)
            if (botResponse.stop) {
                document.getElementById("user-input").disabled = true;
                document.getElementById("send-btn").disabled = true;
            }
        }
    };
    xhr.send("user_input=" + encodeURIComponent(userInput));
}

// Function to check if "Enter" is pressed
function checkEnter(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

// Function to reset the chat
function resetChat() {
    var chatBox = document.getElementById("chat-box");
    chatBox.innerHTML = '';  // Clear the chat history

    // Enable input fields
    document.getElementById("user-input").disabled = false;
    document.getElementById("send-btn").disabled = false;

    // Reload the page or trigger a new conversation
    window.onload();
}
