import json
import random
import nltk
import sqlite3
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Download NLTK data for tokenization and lemmatization
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Load the intents file containing predefined intents and responses
with open('car_intents.json') as file:
    intents = json.load(file)

# Preprocess data
words = []  # Stores all words from patterns
classes = []  # Stores all intent tags
documents = []  # Stores (pattern, tag) pairs
ignore_chars = ['?', '!', '.', ',']  # Characters to ignore

# Tokenize and process each pattern in intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize pattern into words
        words.extend(word_list)  # Add words to words list
        documents.append((' '.join(word_list), intent['tag']))  # Store pattern and tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add unique tags to classes list

# Lemmatize and clean words (remove duplicates and sort them)
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
words = sorted(list(set(words)))  # Remove duplicates and sort words
classes = sorted(list(set(classes)))  # Sort the intent tags

# Prepare training data for machine learning model
X_train = []  # Stores patterns (text)
y_train = []  # Stores corresponding intent tags

for doc in documents:
    X_train.append(doc[0])  # Add pattern text
    y_train.append(doc[1])  # Add corresponding tag

# Create a machine learning pipeline with TF-IDF vectorization and SVM classifier
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))

# Train the model using prepared training data
model.fit(X_train, y_train)

# Function to fetch car details from the database
def get_car_details(brand, model):
    """Fetch price, engine, and features of a car from the database."""
    conn = sqlite3.connect('car_database.db')  # Connect to the database
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT price, engine, features FROM cars
    WHERE brand = ? AND model = ?
    ''', (brand, model))  # Query the database for car details
    
    result = cursor.fetchone()  # Fetch the first matching record
    conn.close()  # Close the database connection
    
    if result:
        return {
            "price": result[0],
            "engine": result[1],
            "features": result[2].split(", ")  # Convert features string into a list
        }
    else:
        return None  # Return None if no details found

# Function to fetch available car brands from the database
def get_available_brands():
    """Fetch all distinct car brands from the database."""
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT DISTINCT brand FROM cars
    ''')  # Query to get unique car brands
    
    brands = [row[0] for row in cursor.fetchall()]  # Extract brand names
    conn.close()
    
    return brands  # Return the list of brands

# Function to fetch available car models for a given brand
def get_available_models(brand):
    """Fetch all available models for a given car brand."""
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT model FROM cars
    WHERE brand = ?
    ''', (brand,))  # Query to get models for the selected brand
    
    models = [row[0] for row in cursor.fetchall()]  # Extract model names
    conn.close()
    
    return models  # Return the list of models

# Chat function to interact with the user
def chat():
    print("Bot is running! Type 'quit' to exit")  # Display startup message
    context = None  # Keep track of the conversation context
    current_brand = None  # Store the selected car brand

    while True:
        inp = input("You: ")  # Get user input
        if inp.lower() == 'quit':
            break  # Exit the chat if the user types 'quit'

        # Process user input: Tokenize and lemmatize words
        inp_words = nltk.word_tokenize(inp)
        inp_words = [lemmatizer.lemmatize(word.lower()) for word in inp_words]
        processed_input = ' '.join(inp_words)

        # Predict intent using the trained model
        predicted_tag = model.predict([processed_input])[0]

        # Handle intent for fetching car information
        if predicted_tag == "car_info":
            context = "brand_selection"  # Set context to brand selection
            available_brands = get_available_brands()  # Get available car brands
            print(f"Which car brand are you interested in? Available brands: {', '.join(available_brands)}")
            continue  # Continue to next input

        # Handle brand selection context
        if context == "brand_selection":
            selected_brand = None
            available_brands = get_available_brands()
            for word in inp_words:
                if word.capitalize() in available_brands:
                    selected_brand = word.capitalize()  # Match brand name
                    break

            if selected_brand:
                context = "model_selection"  # Move to model selection
                current_brand = selected_brand
                available_models = get_available_models(current_brand)  # Get available models
                print(f"Which {current_brand} model are you interested in? Available models: {', '.join(available_models)}")
            else:
                print(f"Sorry, I don't recognize that brand. Please choose from: {', '.join(available_brands)}")
            continue

        # Handle model selection context
        if context == "model_selection":
            car_model = None
            available_models = get_available_models(current_brand)
            for word in inp_words:
                if word.capitalize() in available_models:
                    car_model = word.capitalize()  # Match model name
                    break

            if car_model:
                car_details = get_car_details(current_brand, car_model)  # Fetch car details
                if car_details:
                    print(f"Here are the details for {current_brand} {car_model}:")
                    print(f"Price: {car_details['price']}")
                    print(f"Engine: {car_details['engine']}")
                    print(f"Features: {', '.join(car_details['features'])}")
                else:
                    print("Sorry, I couldn't find details for that car.")
                
                # Ask if the user wants to check another car
                print("Would you like to check another car? (yes/no)")
                context = "ask_another_car"
            else:
                print(f"Sorry, I don't recognize that {current_brand} model. Please choose from: {', '.join(available_models)}")
            continue

        # Handle asking if the user wants to check another car
        if context == "ask_another_car":
            if inp.lower() in ["yes", "y"]:
                context = "brand_selection"  # Reset to brand selection
                available_brands = get_available_brands()
                print(f"Which car brand are you interested in? Available brands: {', '.join(available_brands)}")
            elif inp.lower() in ["no", "n"]:
                context = None  # Reset context
                print("Okay! Let me know if you need anything else.")
            else:
                print("Please answer with 'yes' or 'no'.")
            continue

        # Provide a response for general conversation
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                responses = intent['responses']
                print(random.choice(responses))  # Respond randomly from predefined responses
                break

# Start the chatbot
chat()
