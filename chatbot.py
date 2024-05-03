import streamlit as st
import google.generativeai as genai

f = open("keys\gemini_api_key.txt")
key = f.read()
genai.configure(api_key=key)
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
st.title(":blue[AI Chatbot] :sunglasses:🤖📊")
if "memory1" not in st.session_state:
    st.session_state["memory1"]=[]

chat1=model.start_chat(history=st.session_state["memory1"])

for msg in chat1.history:
       st.chat_message(msg.role).write(msg.parts[0].text)
user_prompt=st.chat_input()

if user_prompt:
        st.chat_message("user").write(user_prompt)
        response = chat1.send_message(user_prompt) 
        for chunk in response:
            st.chat_message("ai").write(chunk.text)   
               
         
        st.session_state["memory1"]=chat1.history
