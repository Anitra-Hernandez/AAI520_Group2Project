# Advanced Generative Chatbot Design Using Cornell Movie Dialogues Corpus - AAI520_Group2Project 

This project is a part of the AAI-520 Natural Language Processing and GenAI course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

### -- Project Status: Completed

# Installation
Follow these steps to install the necessary dependencies and run the project:

### 1. Clone the repository:

git init

git clone https://github.com/Anitra-Hernandez/AAI520_Group2Project.git

cd AAI520_Group2Project

### 2. Install the required Python libraries:

pip install torch transformers nltk os scikit-learn tqdm

### 3. Download the Cornell Movie Dialogues dataset and place the files (movie_lines.txt and movie_conversations.txt) in a dataset/ directory.

### 4. Run the application:

python app.py

### 5. Usage

Once the setup is complete, you can start interacting with the chatbot through a command-line interface. Simply run app.py, and the chatbot will begin responding to your inputs.

Example usage:

You: Hello, how are you?

Bot: I'm doing great! How can I assist you today?

To exit the conversation, type "exit", "quit", or "goodbye".

# Project Intro/Objective
This focus of this project is to build a generative-based chatbot using state-of-the-art deep learning techniques. The chatbot is trained on the Cornell Movie Dialogues Corpus, a rich dataset of movie character dialogues, and leverages Transformer-based architectures to carry out multi-turn conversations, adapt to context, and handle a wide variety of topics.

The model is fine-tuned using GPT-2 (medium), with specific techniques to enhance coherence and relevance in extended conversations. Additionally, the chatbot is capable of filtering inappropriate content, ensuring safe and responsible interaction.

The goal of this project is to design, implement, and evaluate a chatbot that can:
- Engage in multi-turn conversations
- Adapt contextually to ongoing dialogue
- Filter out harmful or inappropriate content
- Provide responses through a user-friendly interface

## Partners/Contributers
- Mohammad Alkhawaldeh - https://www.linkedin.com/in/mohammad-alkhawaldeh/
- Anitra Hernandez - https://www.linkedin.com/in/anitra-jocelyn-hernandez-csm-cspo-b3416188/
- Peter Ogunrinde - https://www.linkedin.com/in/pogunrinde/

## Methods Used
Natural Language Processing (NLP):
- Tokenization: Using the GPT-2 tokenizer to break down text into tokens.
- Text Preprocessing: Preprocessing dialogues by cleaning and structuring the conversation into a format suitable for training.

Generative Modeling:
 -GPT-2 Fine-Tuning: The chatbot is fine-tuned using GPT-2 (medium), a transformer-based model, to generate responses based on dialogue history.

Sequence Modeling:
- Multi-turn conversations: Managing multi-turn conversations by feeding a context of previous exchanges (up to 5 utterances) into the model to generate context-aware responses.

Content Filtering:
 -Regular Expression (Regex) Filtering: Using regex to identify and filter out harmful, offensive, or inappropriate content from both user inputs and chatbot responses.

Evaluation Metrics:
- Perplexity: Used to evaluate the fluency and predictability of the chatbot’s language generation.
- BLEU Score: A metric to evaluate how closely the chatbot's generated responses match the reference dialogues.

Data Splitting:
- Train/Validation/Test Split: Using train_test_split from Scikit-learn to split the dataset into training, validation, and test sets (80/20 split for training and test, further split validation from training).

Training Optimization:
- AdamW Optimizer: Used to optimize the model’s weights during training.
- Learning Rate Scheduling: Implemented with linear warmup using the get_linear_schedule_with_warmup method to adjust the learning rate during training.

## Technologies
- Python
- Jupyter Notebook
- Github

# Project Description

## Dataset
We used the Cornell Movie Dialogues Corpus to train the chatbot. This dataset includes over 220,000 conversational exchanges between characters from 617 movies, providing a diverse set of dialogue patterns for the chatbot to learn from.

### Preprocessing Steps:
- Tokenization: Used the GPT-2 tokenizer to preprocess the movie dialogues.
- Padding and Truncation: Sequences were padded to ensure uniform length, with padding added to the left for GPT-2 compatibility.

## Model Architecture
Our chatbot utilizes GPT-2 (medium), a Transformer-based model from OpenAI, fine-tuned on the movie dialogues for dialogue generation. The following techniques were employed during model training:

- Max History: We use up to 5 previous exchanges to provide the model with context for generating responses.
- Truncation and Padding: Input sequences are truncated to fit within a 512-token limit to optimize memory usage.
- Content Filtering: Regex-based filtering was implemented to prevent the model from generating inappropriate content, such as profanity, violence, or harmful speech.

## Training
### Model Fine-Tuning
The GPT-2 model was fine-tuned on the Cornell Movie Dialogues Corpus for 3 epochs using the AdamW optimizer with a learning rate of 3e-5. The training process involved the following steps:

- Batch Size: 4
- Learning Rate: 3e-5 with linear warmup
- Evaluation Metric: Perplexity and BLEU scores

### Evaluation Metrics
- Perplexity: A lower perplexity indicates that the model predicts the next word in a sequence with more certainty.
- BLEU Score: We used the BLEU metric to evaluate the quality of responses by comparing them to reference dialogues.

## Results
### Validation Reuslts: 
- Perplexity: Achieved a validation perplexity of 11.5412, indicating relatively fluent and coherent text generation.
- BLEU Score: Achieved an average BLEU score of 0.0123, indicating a poor match between generated and reference responses.

### Test Results:
- Perplexity on Test Set: 11.4023

## Content Filtering
We implemented content filtering to ensure that the chatbot avoids generating harmful, inappropriate, or offensive content. The filtering is regex-based, targeting patterns related to profanity, explicit content, and violence.

## Limitations and Future Work
### Current Limitations
- The chatbot occasionally generates irrelevant responses during long dialogues, particularly when context shifts abruptly.
- The filtering system, while effective, could be improved with more sophisticated techniques like sentiment analysis or toxicity detection.
### Future Work
- Fine-tuning with reinforcement learning could further improve the relevance and coherence of the responses.
- Web Interface: Developing a web-based front-end for more user-friendly interaction.
- Improved Content Filtering using machine learning-based methods for detecting harmful or offensive content.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.

# Acknowledgments

We would like to thank:

- Professor Kahila Mokhtari Jahid and The University of San Diego for providing this course.
- OpenAI for making GPT-2 publicly available.
- The creators of the Cornell Movie Dialogues Corpus for the dataset.
