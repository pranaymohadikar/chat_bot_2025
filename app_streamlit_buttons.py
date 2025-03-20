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
#added onn 20-3-25
import uuid
import datetime



# Page configuration
st.set_page_config(page_title="Car Information Chatbot", page_icon="🚗")
st.title("🚗 Car Information Chatbot")

#===============20-3-25====================================#
db_name = "chatbot.db"

class ChatbotSession:
    def __init__(self):
        #initialized the chatbot session
        self.session_id = str(uuid.uuid4()) #generate unique session id
        self.user_name = None
        self.start_time =datetime.datetime.now()
        self.context = None
        self.store_session()
        
    def store_session(self):
        #save session start details in the db
        
        conn = sqlite3.connect(db_name)
        cursor=conn.cursor()
        cursor.execute('''insert into chat_sessions (
            session_id, start_time
            ) values(?, ?)''', (self.session_id, self.start_time))
        conn.commit()
        conn.close()
        
    def update_user_name(self, user_name):
        self.user_name = user_name
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE chat_sessions SET user_name = ? WHERE session_id = ?
        ''', (self.user_name, self.session_id))
        conn.commit()
        conn.close()
        
        
        
    def close_session(self):
        "update the session_endtime"
        conn= sqlite3.connect(db_name)
        cursor=conn.cursor()
        end_time = datetime.datetime.now()
        cursor.execute('''
                       update chat_sessions set end_time = ? where
                       session_id = ?
                       ''', (end_time, self.session_id))
        conn.commit()
        conn.close()
        print("session_closed")
        
#start session

if "chatbot_session" not in st.session_state:
    st.session_state.chatbot_session = ChatbotSession()
#===================================================================
    

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

#print(type(model))

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
    cursor.execute('SELECT distinct model FROM cars WHERE lower(brand )= lower(?)', (brand,))
    models = [row[0] for row in cursor.fetchall()]
    conn.close()
    return models

def get_car_details(brand, model):
    conn = sqlite3.connect('car_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price, engine, features FROM cars WHERE lower(brand) = lower(?) AND lower(model) = lower(?)', (brand, model))
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
#========19-3-25=====#
#username added
if "username" not in st.session_state:
    st.session_state.username = None
    
if "context" not in st.session_state:
    st.session_state.context ='name_request'  # Start by asking for the user's name
    

#=============#19-3-25=======================#
    
    
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! What's your name?"}]
    #=============#19-3-25=======================#

# if 'context' not in st.session_state:
#     st.session_state.context = None 

if 'current_brand' not in st.session_state:
    st.session_state.current_brand = None

#=========20 3 25 chnages for left and right alignment================


# Display messages
for message in st.session_state.messages:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(f"<div style='text-align: right;'>{message['content']}</div>", unsafe_allow_html=True) 
    else:
        with st.chat_message("assistant"):
            st.markdown(f"<div style='text-align: left;'>{message['content']}</div>", unsafe_allow_html=True)
            #st.write(message["content"])
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# Process user input
def process_user_input(user_input):
    if not user_input:
        return
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    user_input = user_input.title()
    
        #==================19-3-25================================#  
    if st.session_state.context == "name_request":
        st.session_state.username = user_input  # Store the name
        #=================20-3-25=================================
        #st.session_state.chatbot_session.store_session() #to start the session
        
        st.session_state.chatbot_session.update_user_name(user_input) # update session with the user name
        #===================================================================
        
        response = f"Nice to meet you, {user_input}! How can I assist you with car information?"
        st.session_state.context = None # added to dont go to the brand selection
        # st.session_state.context = "brand_selection"  # Move to the next step
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    #==================19-3-25================================#

    # Entity extraction
    entities = extract_entities(user_input)
    brand = entities.get("BRAND", "").capitalize()
    model_entity = entities.get("MODEL", "").capitalize()
    print(f'extracted entities: {entities}')

    # Tokenize and classify intent
    processed_input = ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)])
    predicted_tag = model.predict([processed_input])[0]
    
    #added 18-3-2025 21:53
    print(processed_input)
    print(predicted_tag)
    



    #handle name request
    
  
    response = ""
    
#==================19-3-25===============================#
    # changes for the both brand and model entity 
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
            
#==================19-3-25===============================#   

    # elif predicted_tag.endswith("_info") or st.session_state.context == "model_selection":
    #     if brand:
    #         st.session_state.current_brand = brand  # Ensure brand is stored correctly
    #         st.session_state.context = "model_selection"
    #         response = f"Great! Now, please choose a model for {brand}."
            
    #     elif st.session_state.current_brand and model_entity:
    #         # User already selected a brand, now entering a model
    #         car_details = get_car_details(st.session_state.current_brand, model_entity)
    #         if car_details:
    #             response = f"### {st.session_state.current_brand} {model_entity} Details:\n"
    #             response += f"**Price:** {car_details['price']}\n"
    #             response += f"**Engine:** {car_details['engine']}\n"
    #             response += f"**Features:** {', '.join(car_details['features'])}\n\n"
    #             response += "Would you like to check another car?"
    #             st.session_state.context = "ask_another_car"
    #         else:
    #             response = f"Sorry, no details found for {st.session_state.current_brand} {model_entity}. Try another model?"
    #     else:
    #         response = "I didn't get that. Can you specify a brand or model?"
         
    elif predicted_tag.endswith("_info"):
        if brand and model_entity:
            car_details = get_car_details(brand, model_entity)
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
        elif brand:
            st.session_state.context = "model_selection"
            st.session_state.current_brand = brand
        else:
            st.session_state.context = "brand_selection"
    
    elif st.session_state.context == "model_selection":
        if model_entity:
            
            car_details = get_car_details(st.session_state.current_brand, model_entity)
            if car_details:
                response = f"### {st.session_state.current_brand} {model_entity} Details:\n"
                response += f"**Price:** {car_details['price']}\n"
                response += f"**Engine:** {car_details['engine']}\n"
                response += f"**Features:** {', '.join(car_details['features'])}\n\n"
                response += "Would you like to check another car?"
                st.session_state.context = "ask_another_car"
            else:
                response = f"Sorry, no details for {model_entity}. Try another?"
    
    elif st.session_state.context == "ask_another_car":
        if user_input.lower() in ["yes", "y"]:
            st.session_state.context = "brand_selection"
        elif user_input.lower() in ["no", "n"]:
            st.session_state.context = None
            response = "Okay! Let me know if you need anything else."
            #====================20-3-25=====================
            st.session_state.chatbot_session.close_session()  # Close the session
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
if st.session_state.context == "brand_selection" :
    st.write("### Select a car brand:")
    brands = get_available_brands()
    cols = st.columns(len(brands))  # Create buttons in a row
    for i, brand in enumerate(brands):
        if cols[i].button(brand):
            #need to copy the content of the button changed on 18-3-25
            st.session_state.messages.append({"role": "user", "content": brand})
            st.session_state.current_brand = brand
            st.session_state.context = "model_selection"
            st.rerun()

# **Model Selection UI**
if st.session_state.context == "model_selection":
    st.write(f"### Select a model for {st.session_state.current_brand}:")
    models = get_available_models(st.session_state.current_brand)
    cols = st.columns(len(models))
    for i, j in enumerate(models):
        if cols[i].button(j):
            #need to copy the content of the button changed on 18-3-25
            
            #added this on 20-3-25
            #user_input = f"{st.session_state.current_brand}{j}"
            st.session_state.messages.append({"role": "user", "content": j})
            #process_user_input(user_input)
            process_user_input(f"{st.session_state.current_brand} {j}")
            st.rerun()
#==================19-3-25================================#             
# if st.session_state.context == "name_request":
#     st.session_state.messages.append({"role": "assistant", "content": "Hello! What's your name?"})
#     st.session_state.context = "waiting_for_name"
#     st.rerun()
#==================19-3-25================================# 
# **Chat Input**
user_input = st.chat_input("Type your message here...")
if user_input:
    process_user_input(user_input)
    st.rerun()
