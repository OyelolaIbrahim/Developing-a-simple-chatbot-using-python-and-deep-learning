# CloudyML Chatbot — Deep Learning Chatbot with Flask Web Interface

A conversational AI chatbot built from scratch using Python, 
a Deep Learning neural network, and deployed as a live 
web application using Flask. The chatbot understands 
natural language input and responds intelligently 
across 55+ intent categories.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![Deployment](https://img.shields.io/badge/Deployment-Flask-green)
![NLP](https://img.shields.io/badge/NLP-NLTK-yellow)
![Type](https://img.shields.io/badge/Type-Intent%20Classification-purple)

---

## Overview

Builds and deploys an end-to-end conversational chatbot 
using a Deep Learning neural network trained on a custom 
intents dataset. The model classifies user input into 
one of 55+ intent categories and returns a contextually 
appropriate response. The complete pipeline covers:
text preprocessing with NLTK lemmatisation, Bag-of-Words 
feature extraction, a 3-layer Dense Neural Network trained 
with SGD and Nesterov momentum, model persistence using 
joblib pickle and Keras `.h5`, and a live Flask web 
application serving the chatbot through a browser interface.

---

## Problem Statement

Building a rule-based chatbot requires manually writing 
every possible response pattern — a brittle and unscalable 
approach. This project demonstrates a deep learning 
alternative: train a neural network to classify the 
*intent* behind any user message, then select an 
appropriate response from a predefined response pool. 
This makes the chatbot flexible, extensible, and 
capable of handling natural language variations 
it has never seen before.

---

## Dataset — intents.json

The chatbot's knowledge base is defined in `intents.json` — 
a custom JSON file containing 55+ intent categories, each 
with a set of training patterns and corresponding responses.

- **Format:** JSON
- **Structure:** Each intent has three fields:
  - `tag` — the intent category name
  - `patterns` — example user messages that trigger this intent
  - `responses` — possible chatbot replies (one is chosen randomly)
- **No external download required** — `intents.json` 
  is included in this repository

**Sample intent categories covered:**

| Category | Example User Input | Chatbot Response |
|----------|--------------------|-----------------|
| Greetings | "hi there", "hello" | "hello thanks for checking in" |
| Goodbye | "bye", "good bye" | "have a nice time, welcome back again" |
| Name | "what's your name?", "who are you?" | "I'm a CloudyMLChatBot" |
| AI | "What is AI?" | Definition of Artificial Intelligence |
| Programming | "What is your favorite programming language?" | "Python is the best language..." |
| Robots | "Are you stupid?", "Robots should die" | Witty robot persona responses |
| Chatbot | "What is a chat robot?" | Explanation of chatbots |
| Capabilities | "what can you do for me?" | List of services offered |
| Upcoming Events | "what are the upcoming events?" | Event status response |
| ... | 55+ categories total | ... |


## Approach

### Step 1 — Text Preprocessing (train.py)

- Loaded and parsed `intents.json` to extract all 
  training patterns and their intent tags
- Tokenised each pattern using `nltk.word_tokenize()`
- Applied **WordNet Lemmatisation** using 
  `nltk.stem.WordNetLemmatizer` — converts words 
  to their root form (e.g. "running" → "run")
- Removed punctuation tokens (`!`, `?`)
- Built sorted vocabulary (`words`) and 
  sorted intent list (`classes`)
- Serialised both to disk:
```python
  pickle.dump(words, open('words.pkl', 'wb'))
  pickle.dump(classes, open('classes.pkl', 'wb'))
```

### Step 2 — Feature Extraction (Bag of Words)

For each training pattern, created a binary 
Bag-of-Words vector:
- Length = total vocabulary size
- Value = `1` if word is present in the pattern, 
  `0` if absent

```python
bag = [1 if w in pattern_words else 0 for w in words]
```

Output vectors are one-hot encoded over the 
full list of intent classes.

### Step 3 — Neural Network Architecture

A 3-layer Dense Neural Network built with 
TensorFlow/Keras Sequential API:

| Layer | Type | Units | Activation | Regularisation |
|-------|------|-------|------------|----------------|
| 1 | Dense (input) | 128 | ReLU | Dropout(0.5) |
| 2 | Dense (hidden) | 64 | ReLU | Dropout(0.5) |
| 3 | Dense (output) | N intents | Softmax | — |

```python
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), 
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
```

### Step 4 — Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | Categorical Cross-Entropy |
| Optimiser | SGD with Nesterov Momentum |
| Learning Rate | 0.01 |
| Momentum | 0.9 |
| Decay | 1e-6 |
| Epochs | 150 |
| Batch Size | 5 |

```python
sgd = SGD(learning_rate=0.01, decay=1e-6, 
          momentum=0.9, nesterov=True)
```

Nesterov Accelerated Gradient (NAG) was chosen 
over standard SGD because it "looks ahead" in 
the gradient direction, resulting in faster 
convergence and better performance on 
classification tasks.

### Step 5 — Saving the Model

```python
model.save('chatbot_model.h5', hist)
```

The trained model, vocabulary (`words.pkl`), 
and intent classes (`classes.pkl`) are all 
persisted to disk — loaded at runtime by 
the Flask app without retraining.

### Step 6 — Flask Web Application (app.py)

The Flask app exposes two routes:

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Renders the chatbot HTML interface |
| `/get` | POST | Receives user message, returns chatbot response |

**Prediction pipeline (per user message):**

**Error threshold:**
```python
ERROR_THRESHOLD = 0.25
results = [[i, r] for i, r in enumerate(res) 
           if r > ERROR_THRESHOLD]
```
Predictions below 25% confidence are discarded — 
preventing the chatbot from giving a low-confidence 
wrong answer.

---

## Results

| Metric | Value |
|--------|-------|
| Training Epochs | 150 |
| Batch Size | 5 |
| Intent Categories | 55+ |
---

## Technologies Used

- **Language:** Python 3
- **Deep Learning:** TensorFlow, Keras 
  (Sequential, Dense, Dropout)
- **NLP:** NLTK (word_tokenize, WordNetLemmatizer, 
  stopwords)
- **Web Framework:** Flask
- **Data Serialisation:** Pickle, JSON
- **Optimiser:** SGD with Nesterov Momentum
- **Frontend:** HTML, CSS, JavaScript 
  (static/ and templates/)

---

## How to Run

### Train the model (first time only)

```bash
# 1. Clone the repository
git clone https://github.com/OyelolaIbrahim/chatbot-deep-learning-python.git
cd chatbot-deep-learning-python

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
#    This generates chatbot_model.h5, words.pkl, classes.pkl
python train.py
```

### Launch the web application

```bash
# 4. Start the Flask server
python app.py

# 5. Open your browser and go to:
#    http://127.0.0.1:5000
```


## Extending the Chatbot

To add new topics, edit `intents.json`:

```json
{
  "tag": "your_new_topic",
  "patterns": [
    "example question 1",
    "example question 2"
  ],
  "responses": [
    "Your chatbot response here"
  ]
}
```
