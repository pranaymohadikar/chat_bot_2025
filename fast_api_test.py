from fastapi import FastAPI
import json
import sqlite3
app = FastAPI()

# food_items = {
#     'indian':['samosa','dosa'],
#     'american':['hot dog','apple pie'],
#     'italian':['ravioli','pizza']
# }

# @app.get("/get_items/{cusine}")
# async def get_items(cusine):
#     return food_items.get(cusine)


def get_db_connection():
    return sqlite3.connect('car_database.db')


# API to get all available car brands
@app.get("/brands")
def get_available_brands():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("select brand from cars")
    brands = [row[0] for row in cursor.fetchall()]
    conn.close()
    return {"brands": brands}

@app.get("/models")
def get_available_models(brand):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("select distinct model from cars")
    models = [row[0] for row in cursor.fetchall()]
    conn.close()
    return {'models': models}


@app.get("/cars/{brand}/{model}")
def get_car_details():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("select price, engine, features from cars where brand =? and model = ?",(brand, model))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return{
            "price": result[0],
            'engine':result[1],
            'features':result[2].split(', ')
        }
        
    else:
        return {'car details not found'}
        