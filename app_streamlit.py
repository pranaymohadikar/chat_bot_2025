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
st.title("Car Information Chatbot")

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

# Load models and data (cached so it only runs once)
@st.cache_resource
def load_models_and_data():
    # Load intents file
    with open('car_intents.json') as file:
        intents = json.load(file)
    
    # Load the custom entity recognition model
    nlp = spacy.load("custom_entity_model")
    
    # Preprocess data
    words = []
    classes = []
    documents = []
    ignore_chars = ['?', '!', '.', ',']
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize words
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((' '.join(word_list), intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    # Lemmatize and clean words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    
    # Prepare training data
    X_train = []  # Patterns
    y_train = []  # Intents (tags)
    
    for doc in documents:
        X_train.append(doc[0])  # Pattern
        y_train.append(doc[1])  # Tag
    
    # Verify training data
    if not X_train or not y_train:
        raise ValueError("Training data is empty. Check your intents.json file.")
    
    # Create a pipeline with TF-IDF and SVM
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model, nlp, intents

model, nlp, intents = load_models_and_data()

# Function to fetch car details from the database
def get_car_details(brand, model):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT price, engine, features FROM cars
    WHERE brand = ? AND model = ?
    ''', (brand, model))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            "price": result[0],
            "engine": result[1],
            "features": result[2].split(", ")
        }
    else:
        return None

# Function to fetch available brands from the database
def get_available_brands():
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    
    # Fetch all distinct car brands from the database
    cursor.execute('SELECT DISTINCT brand FROM cars')
    brands = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return brands

# Function to fetch available models for a brand
def get_available_models(brand):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    
    # Fetch all models for the given brand
    cursor.execute('SELECT distinct model FROM cars WHERE brand = ?', (brand,)) #added distinct keyword on 18/3/25
    models = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return models

# Function to extract entities from user input
def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    return entities

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your car information assistant. How can I help you today?"})

if 'context' not in st.session_state:
    st.session_state.context = None

if 'current_brand' not in st.session_state:
    st.session_state.current_brand = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to process user input and generate responses
def process_user_input(user_input):
    if not user_input:
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process the user's message
    user_input = user_input.title()  # Convert input to title case
    entities = extract_entities(user_input)
    
    brand = entities.get("BRAND", "").capitalize()
    model_entity = entities.get("MODEL", "")
    
    # Debug info that can be shown if needed
    debug_info = f"Detected Brand: {brand}, Model: {model_entity}"
    
    # Tokenize and lemmatize input for intent classification
    inp_words = nltk.word_tokenize(user_input)
    inp_words = [lemmatizer.lemmatize(word.lower()) for word in inp_words if word not in ['?', '!', ',']]
    processed_input = ' '.join(inp_words)
    
    predicted_tag = model.predict([processed_input])[0]
    
    response = ""
    
    # Handle brand & model-based car information
    if predicted_tag.endswith("_info"):
        if brand and model_entity:
            car_details = get_car_details(brand, model_entity)
            if car_details:
                response = f"### Car Details for {brand} {model_entity}:\n"
                response += f"**Price:** {car_details['price']}\n"
                response += f"**Engine:** {car_details['engine']}\n"
                response += f"**Features:** {', '.join(car_details['features'])}\n\n"
                response += "Would you like to check another car?"
            else:
                response = f"No information found for {brand} {model_entity}. Would you like to check another car?"
            st.session_state.context = "ask_another_car"
        elif brand:
            st.session_state.context = "model_selection"
            st.session_state.current_brand = brand
            available_models = get_available_models(brand)
            response = f"Which {brand} model are you interested in? Available models: {', '.join(available_models)}"
        else:
            st.session_state.context = "brand_selection"
            available_brands = get_available_brands()
            response = f"Which car brand are you interested in? Available brands: {', '.join(available_brands)}"
    
    # Handle model selection context
    elif st.session_state.context == "model_selection":
        if model_entity:
            car_details = get_car_details(st.session_state.current_brand, model_entity)
            if car_details:
                response = f"### Details for {st.session_state.current_brand} {model_entity}:\n"
                response += f"**Price:** {car_details['price']}\n"
                response += f"**Engine:** {car_details['engine']}\n"
                response += f"**Features:** {', '.join(car_details['features'])}\n\n"
                response += "Would you like to check another car?"
            else:
                response = f"Sorry, I couldn't find details for {model_entity}."
                response += "Would you like to check another car?"
            st.session_state.context = "ask_another_car"
        else:
            available_models = get_available_models(st.session_state.current_brand)
            response = f"Sorry, I don't recognize that model. Please choose from: {', '.join(available_models)}"
    
    # Handle asking for another car
    elif st.session_state.context == "ask_another_car":
        if user_input.lower() in ["yes", "y"]:
            st.session_state.context = "brand_selection"
            available_brands = get_available_brands()
            response = f"Which car brand are you interested in? Available brands: {', '.join(available_brands)}"
        elif user_input.lower() in ["no", "n"]:
            st.session_state.context = None
            response = "Okay! Let me know if you need anything else."
        else:
            response = "Please answer with 'yes' or 'no'."
    
    # Default response from intents
    else:
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                response = random.choice(intent['responses'])
                break
        
        # If no response was set, use a fallback
        if not response:
            response = "I'm not sure how to respond to that. Can you try rephrasing?"
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Chat input
user_input = st.chat_input("Type your message here...")
if user_input:
    process_user_input(user_input)
    st.rerun()  # Rerun the app to update the chat history

# Add some helpful information at the bottom
with st.expander("How to use this chatbot"):
    st.write("""
    - Ask about car brands and models to get detailed information
    - You can ask questions like "Tell me about Tesla Model 3" or "What features does Toyota Camry have?"
    - If you don't specify a brand or model, the chatbot will guide you through the selection process
    - You can also ask general questions about cars
    """)