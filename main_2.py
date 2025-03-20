import json
import random
import nltk
import sqlite3
import spacy
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Download NLTK data (only runs once)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load models and data
def load_models_and_data():
    with open('car_intents.json') as file:
        intents = json.load(file)

    nlp = spacy.load("custom_entity_model")
    
    words, classes, documents = [], [], []
    ignore_chars = ['?', '!', '.', ',']
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((' '.join(word_list), intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    words = sorted(set([lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]))
    classes = sorted(set(classes))

    X_train, y_train = [doc[0] for doc in documents], [doc[1] for doc in documents]
    
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    model.fit(X_train, y_train)
    
    return model, nlp, intents

model, nlp, intents = load_models_and_data()

# Database functions
def get_car_details(brand, model):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price, engine, features FROM cars WHERE lower(brand) = lower(?) AND lower(model) = lower(?)', (brand, model))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {"price": result[0], "engine": result[1], "features": result[2].split(", ")}
    return None

# Extract entities from text
def extract_entities(text):
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

# Chatbot session state
session = {
    "username": None,
    "context": "name_request",
    "current_brand": None
}

def process_user_input(user_input):
    if session["context"] == "name_request":
        session["username"] = user_input.title()  # Store the name
        print(f"Nice to meet you, {user_input}! How can I assist you with car information?")
        session["context"] = None
        return
    
    entities = extract_entities(user_input)
    brand = entities.get("BRAND", "").capitalize()
    model_entity = entities.get("MODEL", "").capitalize()
    
    processed_input = ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)])
    predicted_tag = model.predict([processed_input])[0]
    
    response = ""
    
    if brand and model_entity:
        car_details = get_car_details(brand, model_entity)
        if car_details:
            response = f"{brand} {model_entity} Details:\n"
            response += f"Price: {car_details['price']}\n"
            response += f"Engine: {car_details['engine']}\n"
            response += f"Features: {', '.join(car_details['features'])}\n"
        else:
            response = f"No details found for {brand} {model_entity}. Try another one?"
    else:
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                response = random.choice(intent['responses'])
                break
    
        if not response:
            response = "I'm not sure. Can you rephrase?"
    
    print(response)

# Main chat loop
print("Hello! What's your name?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye! Have a great day!")
        break
    process_user_input(user_input)
