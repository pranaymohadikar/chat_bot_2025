import sqlite3
def get_car_details(brand, model):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price, engine, features FROM cars WHERE lower(brand) = lower(?) AND lower(model) = lower(?)', (brand, model))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {"price": result[0], "engine": result[1], "features": result[2].split(", ")}
    return None

def get_available_models(brand):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT distinct lower(model) FROM cars WHERE lower(brand )= lower(?)', (brand,))
    models = [row[0] for row in cursor.fetchall()]
    conn.close()
    return models


def extract_entities(text):
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}


print(extract_entities("tesla model 3"))