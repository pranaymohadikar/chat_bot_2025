import sqlite3
import uuid
import datetime
#==============================20-3-25=============================================
# # create database as chatbot.db

# db_name = "chatbot.db"

# def create_database():
#     "create tables for storing data"
    
#     conn = sqlite3.connect(db_name)
#     cursor = conn.cursor()
    
#     #creating session_table
    
#     cursor.execute('''
#                    create table if not exists chat_sessions (
#                        session_id text primary key,
#                        user_name text,
#                        start_time  datetime,
#                        end_time  datetime
#                    )
                   
#                    ''')    
    
#     conn.commit()
#     conn.close()
#     print("database initialized")
    
# create_database()


# class ChatbotSession:
#     def __init__(self):
#         #initialized the chatbot session
#         self.session_id = str(uuid.uuid4()) #generate unique session id
#         self.user_name = None
#         self.start_time =datetime.datetime.now()
#         self.context = None
#         self.store_session()
        
#     def store_session(self):
#         #save session start details in the db
        
#         conn = sqlite3.connect(db_name)
#         cursor=conn.cursor()
#         Cursor.execute('''insert into chat_sessions (
#             session_id, start_time
#             ) values(?, ?)''', (self.session_id, self.start_time))
#         conn.commit()
#         conn.close()
        
        
#     def close_session(self):
#         "update the session_endtime"
#         conn= sqlite3.connect(db_name)
#         cursor=conn.cursor()
#         end_time = datetime.datetime.now()
#         cursor.execute('''
#                        update chat_sessions set end_time = ? where
#                        session_id = ?
#                        ''', (end_time, self_session_id))
#         conn.commit()
#         conn.close()
#         pritn("session_closed")
        
# #start session

# session = ChatbotSession()
        
        
db_name = "chatbot.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Query the chat_sessions table
cursor.execute("SELECT * FROM chat_sessions")
rows = cursor.fetchall()

# Print the results
if rows:
    print("Session Data:")
    for row in rows:
        print(f"Session ID: {row[0]}, User Name: {row[1]}, Start Time: {row[2]}, End Time: {row[3]}")
else:
    print("No session data found.")

# Close the connection
conn.close()

#==============================20-3-25=============================================