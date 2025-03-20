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
    cursor.execute('SELECT DISTINCT model FROM cars WHERE LOWER(brand) = LOWER(?)', (brand,))
    models = [row[0] for row in cursor.fetchall()]
    conn.close()
    return models

def get_car_details(brand, model):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price, engine, features FROM cars WHERE LOWER(brand) = LOWER(?) AND LOWER(model) = LOWER(?)', (brand, model))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {"price": result[0], "engine": result[1], "features": result[2].split(", ")}
    return None

# Extract entities from text
def extract_entities(text):
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

# Chatbot class
class CarChatbot:
    def __init__(self):
        self.username = None
        self.context = "name_request"  # Start by asking for the user's name
        self.current_brand = None
        self.messages = [{"role": "assistant", "content": "Hello! What's your name?"}]

    def process_user_input(self, user_input):
        if not user_input:
            return
        
        self.messages.append({"role": "user", "content": user_input})
        
        # Handle name request
        if self.context == "name_request":
            self.username = user_input  # Store the name
            response = f"Nice to meet you, {user_input}! How can I assist you with car information?"
            self.context = None  # Move to the next step
            self.messages.append({"role": "assistant", "content": response})
            return response

        # Entity extraction
        entities = extract_entities(user_input)
        brand = entities.get("BRAND", "").capitalize()
        model_entity = entities.get("MODEL", "").capitalize()

        # Tokenize and classify intent
        processed_input = ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)])
        predicted_tag = model.predict([processed_input])[0]

        response = ""

        # Handle both brand and model in input
        if brand and model_entity:
            car_details = get_car_details(brand, model_entity)
            if car_details:
                response = f"### {brand} {model_entity} Details:\n"
                response += f"**Price:** {car_details['price']}\n"
                response += f"**Engine:** {car_details['engine']}\n"
                response += f"**Features:** {', '.join(car_details['features'])}\n\n"
                response += "Would you like to check another car?"
                self.context = "ask_another_car"
            else:
                response = f"No details found for {brand} {model_entity}. Try another one?"
                self.context = "brand_selection"
            self.messages.append({"role": "assistant", "content": response})
            return response

        # Handle model-only input
        elif not brand and model_entity:
            if self.current_brand:  # Use the selected brand
                brand = self.current_brand
                car_details = get_car_details(brand, model_entity)
                if car_details:
                    response = f"### {brand} {model_entity} Details:\n"
                    response += f"**Price:** {car_details['price']}\n"
                    response += f"**Engine:** {car_details['engine']}\n"
                    response += f"**Features:** {', '.join(car_details['features'])}\n\n"
                    response += "Would you like to check another car?"
                    self.context = "ask_another_car"
                else:
                    response = f"No details found for {brand} {model_entity}. Try another one?"
                    self.context = "brand_selection"
            else:  # No brand selected
                response = "Please specify the brand (e.g., 'Toyota Camry')."
            self.messages.append({"role": "assistant", "content": response})
            return response

        # Handle only brand in input
        elif brand:
            self.context = "model_selection"
            self.current_brand = brand
            response = f"Great! Please select a model for {brand}."
            self.messages.append({"role": "assistant", "content": response})
            return response

        # Handle other cases
        else:
            for intent in intents['intents']:
                if intent['tag'] == predicted_tag:
                    response = random.choice(intent['responses'])
                    break
            
            if not response:
                response = "I'm not sure. Can you rephrase?"

        self.messages.append({"role": "assistant", "content": response})
        return response

# Main function to run the chatbot
def main():
    chatbot = CarChatbot()
    print(chatbot.messages[0]["content"])  # Print the initial message

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break

        response = chatbot.process_user_input(user_input)
        print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    main()