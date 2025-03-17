import json
import random
import nltk
import sqlite3
import spacy
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

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

# Verify model initialization
if model is None:
    raise ValueError("Model is not initialized. Check the pipeline creation.")

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
    print(result[0])
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
    cursor.execute('SELECT model FROM cars WHERE brand = ?', (brand,))
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


def chat():
    print("Bot is running! Type 'quit' to exit")
    context = None
    current_brand = None

    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break
        
        inp = inp.title()  # Convert input to title case
        entities = extract_entities(inp)  # Extract entities properly

        brand = entities.get("BRAND", "").capitalize()  # Ensure brand is in proper case
        model_entity = entities.get("MODEL", "")  # Keep model as is (multi-word models)

        print(f"Detected Brand: {brand}")
        print(f"Detected Model: {model_entity}")

        # Tokenize and lemmatize input
        inp_words = nltk.word_tokenize(inp)
        inp_words = [lemmatizer.lemmatize(word.lower()) for word in inp_words if word not in ['?', '!', ',']]
        processed_input = ' '.join(inp_words)

        if model is None:
            print("Error: Model not available.")
            continue

        predicted_tag = model.predict([processed_input])[0]
        
        # Handle brand & model-based car information
        if predicted_tag.endswith("_info"):
            if brand and model_entity:
                car_details = get_car_details(brand, model_entity)
                if car_details:
                    print(f"Car Details for {brand} {model_entity}:")
                    print(f"Price: {car_details['price']}")
                    print(f"Engine: {car_details['engine']}")
                    print(f"Features: {', '.join(car_details['features'])}")
                else:
                    print(f"No information found for {brand} {model_entity}.")
                print("Would you like to check another car? (yes/no)")
                context = "ask_another_car"
            elif brand:
                context = "model_selection"
                current_brand = brand
                available_models = get_available_models(current_brand)
                print(f"Which {current_brand} model are you interested in? Available models: {', '.join(available_models)}")
            else:
                context = "brand_selection"
                available_brands = get_available_brands()
                print(f"Which car brand are you interested in? Available brands: {', '.join(available_brands)}")
            continue

        # Handle model selection
        if context == "model_selection":
            if model_entity:
                car_details = get_car_details(current_brand, model_entity)
                if car_details:
                    print(f"Here are the details for {current_brand} {model_entity}:")
                    print(f"Price: {car_details['price']}")
                    print(f"Engine: {car_details['engine']}")
                    print(f"Features: {', '.join(car_details['features'])}")
                else:
                    print("Sorry, I couldn't find details for that car.")
                
                print("Would you like to check another car? (yes/no)")
                context = "ask_another_car"
            else:
                available_models = get_available_models(current_brand)
                print(f"Sorry, I don't recognize that {current_brand} model. Please choose from: {', '.join(available_models)}")
            continue

        # Ask for another car
        if context == "ask_another_car":
            if inp.lower() in ["yes", "y"]:
                context = "brand_selection"
                available_brands = get_available_brands()
                print(f"Which car brand are you interested in? Available brands: {', '.join(available_brands)}")
            elif inp.lower() in ["no", "n"]:
                context = None
                print("Okay! Let me know if you need anything else.")
            else:
                print("Please answer with 'yes' or 'no'.")
            continue

        # Default response
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                print(random.choice(intent['responses']))
                break
            
            
chat()
