from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import threading
import time
import re

# Initialize the Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_path = 'fine_tuned_gpt2_medium.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Load the fine-tuned weights (adjust based on your model's architecture)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Content filtering to prevent inappropriate responses
disallowed_words = [
    r'self_harm', r'suicide', r'death', r'kill', r'porn', r'nude', r'rape', r'violence', 
    r'gun', r'explosion', r'assault', r'attack', r'mutilation', r'bitch', r'asshole', r'cunt',
    r'\b(assault|fuck|shooting|bomb|explosion|attack|blood|mutilation|massacre)\b'
]
pattern = re.compile(r'\b(' + '|'.join([re.escape(word) for word in disallowed_words]) + r')\b', re.IGNORECASE)

# Function to check for disallowed content
def contains_disallowed_content(text):
    return bool(pattern.search(text))


# Define the Chatbot class
class Chatbot:
    def __init__(self, model, tokenizer, max_history=4):
        self.model = model
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.chat_history = []
        self.device = device
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def reset_history(self):
        self.chat_history = []

    def get_response(self, user_input):
        # Filter inappropriate content
        if contains_disallowed_content(user_input):
            return "I'm sorry, but I cannot assist with that request.", True

        if user_input.lower() in ["exit", "quit", "goodbye"]:
            return "Goodbye! Have a great day.", True  # True indicates to stop the conversation

        self.chat_history.append(user_input)
        history = self.chat_history[-self.max_history:]
        input_text = "<|endoftext|>".join(history) + "<|endoftext|>"

        # Encode the input
        encoding = self.tokenizer(
            input_text, return_tensors='pt', max_length=512, truncation=True, padding=True
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Generate a response
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,  # Generate up to 50 tokens
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,    # Sampling for variety
            top_p=0.85,        # Nucleus sampling to balance quality and diversity
            top_k=40,          # Limit to top-k tokens for faster generation
            temperature=0.7,   # Temperature for more coherent responses
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            repetition_penalty=1.2,    # Penalize repetition
            no_repeat_ngram_size=3     # Prevent repeating n-grams
        )

        # Decode and return the generated text
        generated_tokens = output[0][input_ids.size(-1):]
        reply = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Check if the generated reply contains disallowed content
        if contains_disallowed_content(reply):
            reply = "Sorry, I cannot discuss that topic."

        # Filter the response if it contains disallowed content or seems nonsensical
        #if contains_disallowed_content(reply) or len(reply.strip()) == 0:
            #reply = random.choice(response_templates)

        self.chat_history.append(reply)
        return reply.strip(), False  # False means continue the conversation

# Instantiate the chatbot
chatbot = Chatbot(model, tokenizer)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

# Route to start the conversation with an initial greeting
@app.route('/start_conversation')
def start_conversation():
    response = "Hello! How can I assist you today?"
    return jsonify({'response': response})

# Route for processing user input and returning the chatbot's response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    response, stop_conversation = chatbot.get_response(user_input)
    
    # If the conversation is to stop, return the final message and an indicator
    if stop_conversation:
        chatbot.reset_history()  # Reset conversation history when ending
        return jsonify({'response': response, 'stop': True})
    
    return jsonify({'response': response, 'stop': False})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
