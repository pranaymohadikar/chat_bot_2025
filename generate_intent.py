import sqlite3
import json

# Connect to the SQLite database
conn = sqlite3.connect('car_database.db')
cursor = conn.cursor()

# Fetch all car brands and models from the database
cursor.execute('SELECT DISTINCT brand FROM cars')
brands = [row[0] for row in cursor.fetchall()]

# Generate intents dynamically
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you"],
            "responses": ["Hello!", "Hi there!", "Good to see you!"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "See you later!", "Take care!"]
        },
        {
            "tag": "car_info",
            "patterns": ["Tell me about cars", "I want to know about cars", "Car details"],
            "responses": ["Which car brand are you interested in? (Toyota, Honda, Ford, Tesla)"]
        }
    ]
}

# Add intents for each car brand
for brand in brands:
    intent = {
        "tag": f"{brand.lower()}_info",
        "patterns": [
            f"Tell me about {brand} cars",
            f"I want to know about {brand}",
            f"{brand} car details"
        ],
        "responses": [f"Which {brand} model are you interested in?"]
    }
    intents["intents"].append(intent)

# Save intents to a JSON file
with open('car_intents.json', 'w') as file:
    json.dump(intents, file, indent=4)

print("car_intents.json file generated successfully!")