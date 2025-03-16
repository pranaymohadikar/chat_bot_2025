import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('car_database.db')
cursor = conn.cursor()

# Create a table for car details
cursor.execute('''
CREATE TABLE IF NOT EXISTS cars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    price TEXT NOT NULL,
    engine TEXT NOT NULL,
    features TEXT NOT NULL
)
''')

# Insert sample data
cars_data = [
    ("Toyota", "Corolla", "$20,000", "1.8L 4-cylinder", "Bluetooth, Backup Camera, Apple CarPlay"),
    ("Toyota", "Camry", "$25,000", "2.5L 4-cylinder", "Leather Seats, Sunroof, Android Auto"),
    ("Toyota", "Rav4", "$28,000", "2.5L 4-cylinder", "AWD, Touchscreen Display, Blind Spot Monitor"),
    ("Honda", "Civic", "$22,000", "2.0L 4-cylinder", "Bluetooth, LaneWatch, Apple CarPlay"),
    ("Honda", "Accord", "$27,000", "1.5L Turbo 4-cylinder", "Leather Seats, Sunroof, Android Auto"),
    ("Ford", "Mustang", "$35,000", "5.0L V8", "Leather Seats, Premium Audio, Performance Package"),
    ("Ford", "F-150", "$40,000", "3.5L EcoBoost V6", "Towing Package, Touchscreen Display, Android Auto"),
    ("Tesla", "Model 3", "$45,000", "Electric", "Autopilot, Touchscreen Display, Over-the-Air Updates"),
    ("Tesla", "Model S", "$80,000", "Electric", "Ludicrous Mode, Autopilot, Premium Interior")
]

cursor.executemany('''
INSERT INTO cars (brand, model, price, engine, features)
VALUES (?, ?, ?, ?, ?)
''', cars_data)



cursor.execute('select brand, model from cars')
car_data = cursor.fetchall()
# Commit changes and close the connection
conn.commit()
conn.close()
for brand, model in car_data:
    print(brand, model)
print("Database and table created successfully!")




# Function to generate training data for entity recognition
def generate_training_data(car_data):
    training_data = []
    
    for brand, model in car_data:
        # Create patterns and annotate entities
        pattern1 = f"Tell me about {brand} {model}"
        entities1 = [
            (pattern1.find(brand), pattern1.find(brand) + len(brand), "BRAND"),
            (pattern1.find(model), pattern1.find(model) + len(model), "MODEL")
        ]
        training_data.append((pattern1, {"entities": entities1}))
        
        pattern2 = f"What are the details of {brand} {model}?"
        entities2 = [
            (pattern2.find(brand), pattern2.find(brand) + len(brand), "BRAND"),
            (pattern2.find(model), pattern2.find(model) + len(model), "MODEL")
        ]
        training_data.append((pattern2, {"entities": entities2}))
    
    return training_data

# Generate training data
TRAIN_DATA = generate_training_data(car_data)

# Print training data
for data in TRAIN_DATA:
    print(data)
    
    
import json

# Save TRAIN_DATA to a JSON file
with open("train_data_entity.json", "w") as file:
    json.dump(TRAIN_DATA, file, indent=4)

print("TRAIN_DATA saved to train_data.json")
