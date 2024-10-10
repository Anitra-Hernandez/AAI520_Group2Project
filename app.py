from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

        # Generate a response with optimizations for speed
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,  # Limit token generation to speed up
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            top_p=0.90,      # Reduce top_p for faster, focused sampling
            top_k=30,       # Lower top_k for faster response
            temperature=0.7, # Slightly lower temperature for faster responses
            eos_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            repetition_penalty=1.2,  # Penalize repetition for better response quality
            no_repeat_ngram_size=3
        )

        # Decode and return the generated text
        generated_tokens = output[0][input_ids.size(-1):]
        reply = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
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
        return jsonify({'response': response, 'stop': True})
    
    return jsonify({'response': response, 'stop': False})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)