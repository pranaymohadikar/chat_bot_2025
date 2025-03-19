import json
import random
import nltk
import sqlite3
import spacy
import streamlit as st
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Page configuration
st.set_page_config(page_title="Car Information Chatbot", page_icon="🚗")
st.title("🚗 Car Information Chatbot")

# Download NLTK data (only runs once)
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    return "Downloaded"

download_nltk_data()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load models and data (cached)
@st.cache_resource
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
def get_available_brands():
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT brand FROM cars')
    brands = [row[0] for row in cursor.fetchall()]
    conn.close()
    return brands

def get_available_models(brand):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT model FROM cars WHERE brand = ?', (brand,))
    models = [row[0] for row in cursor.fetchall()]
    conn.close()
    return models

def get_car_details(brand, model):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price, engine, features FROM cars WHERE brand = ? AND model = ?', (brand, model))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {"price": result[0], "engine": result[1], "features": result[2].split(", ")}
    return None

# Extract entities from text
def extract_entities(text):
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

# Session state variables
if "username" not in st.session_state:
    st.session_state.username = None

if "context" not in st.session_state:
    st.session_state.context = 'name_request'  # Start by asking for the user's name

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your car assistant. How can I help?"}]

if 'current_brand' not in st.session_state:
    st.session_state.current_brand = None

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process user input
def process_user_input(user_input):
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if st.session_state.context == "name_request":
        st.session_state.username = user_input  # Store the name
        response = f"Nice to meet you, {user_input}! How can I assist you with car information?"
        st.session_state.context = "brand_selection"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Entity extraction
    entities = extract_entities(user_input)
    brand = entities.get("BRAND", "").capitalize()
    model_entity = entities.get("MODEL", "")
    print(brand)
    print(model_entity)

    # Tokenize and classify intent
    processed_input = ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)])
    predicted_tag = model.predict([processed_input])[0]

    response = ""

    # **Direct Brand & Model Query Handling**
    if brand and model_entity:
        car_details = get_car_details(brand, model_entity)
        print(car_details)
        if car_details:
            response = f"### {brand} {model_entity} Details:\n"
            response += f"**Price:** {car_details['price']}\n"
            response += f"**Engine:** {car_details['engine']}\n"
            response += f"**Features:** {', '.join(car_details['features'])}\n\n"
            response += "Would you like to check another car?"
            st.session_state.context = "ask_another_car"
        else:
            response = f"No details found for {brand} {model_entity}. Try another one?"
            st.session_state.context = "brand_selection"

    elif predicted_tag.endswith("_info"):
        if brand:
            st.session_state.context = "model_selection"
            st.session_state.current_brand = brand
        else:
            st.session_state.context = "brand_selection"

    elif st.session_state.context == "ask_another_car":
        if user_input.lower() in ["yes", "y"]:
            st.session_state.context = "brand_selection"
        elif user_input.lower() in ["no", "n"]:
            st.session_state.context = None
            response = "Okay! Let me know if you need anything else."
        else:
            response = "Please answer 'yes' or 'no'."

    else:
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                response = random.choice(intent['responses'])
                break
        
        if not response:
            response = "I'm not sure. Can you rephrase?"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# **Brand Selection UI**
if st.session_state.context == "brand_selection":
    st.write("### Select a car brand:")
    brands = get_available_brands()
    cols = st.columns(len(brands))
    for i, brand in enumerate(brands):
        if cols[i].button(brand):
            st.session_state.messages.append({"role": "user", "content": brand})
            st.session_state.current_brand = brand
            st.session_state.context = "model_selection"
            st.rerun()

# **Chat Input**
user_input = st.chat_input("Type your message here...")
if user_input:
    process_user_input(user_input)
    st.rerun()
