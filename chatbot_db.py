import sqlite3
import uuid
import datetime
#==============================20-3-25=============================================
# create database as chatbot.db

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
    
#     cursor.execute('''create table if not exists chat_messages (
#         message_id integer primary key autoincrement,
#         session_id text forign key,
#         sender text,
#         message_text text,
#         timestamp datetime,
#         foreign key (session_id) references chat_sessions(session_id))
# ''') 
    
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
#         cursor.execute('''insert into chat_sessions (
#             session_id, start_time
#             ) values(?, ?)''', (self.session_id, self.start_time))
#         conn.commit()
#         conn.close()
        
#     def store_messages(self, sender, message, time):
#         #save the messags from bot and user
#         timestamp = datetime.datetime.now()
#         conn = sqlite3.connect(db_name)
#         cursor = conn.cursor()
#         cursor.execute('''
#                        insert into chat_messages (session_id, sender, message_text, timestamp) values (?,?,?,?)
#                        ''', (self.session_id, sender, message, timestamp))
        
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
        
        
# db_name = "chatbot.db"
# conn = sqlite3.connect(db_name)
# cursor = conn.cursor()

# # Query the chat_sessions table
# cursor.execute("SELECT * FROM chat_sessions")
# rows = cursor.fetchall()
# #print(rows)

# # Print the results
# if rows:
#     print("Session Data:")
#     for row in rows:
#         print(f"Session ID: {row[0]}, User Name: {row[1]}, Start Time: {row[2]}, End Time: {row[3]}")
# else:
#     print("No session data found.")

# # Close the connection
# conn.close()

#==============================20-3-25=============================================

#+==================================22-3-25=====================================
db_name = "chatbot.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Query the chat_sessions table
cursor.execute("SELECT * FROM chat_messages")
messages = cursor.fetchall()
#print(rows)

# Print the results
if messages:
    print("===== Chat Messages =====")
    for message in messages:
        print(f"Message ID: {message[0]}, Session ID: {message[1]}, Sender: {message[2]}, Message: {message[3]}, Timestamp: {message[4]}")
else:
    print("No chat messages found.")

# Close the connection
conn.close()

#===========================================================================================