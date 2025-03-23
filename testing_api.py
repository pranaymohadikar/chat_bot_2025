import requests
import sqlite3

API = "https://www.freetestapi.com/api/v1/cars"

def fetch_car_data():
    response = requests.get(API)
    if response.status_code == 200:
        return response.json()
        return []
    
    
car_data = fetch_car_data()
print(car_data)
    