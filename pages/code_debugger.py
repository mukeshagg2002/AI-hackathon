import streamlit as st
from PIL import Image
import google.generativeai as genai
import io
f = open("keys\gemini_api_key.txt")
key = f.read()
genai.configure(api_key=key)
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest",
                              system_instruction="you are a  Ai code debugger assistant your task is to only identify bugs or errors in the given image code or in the prompt and provide the correct code and the bugs and if someone ask anything beyond programming then say I don't know")
user_prompt=st.chat_input()

st.title("Code Debugger app")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  image_bytes = io.BytesIO()
  image = image.convert("RGB") 
  image.save(image_bytes, format="JPEG")
  image_data = image_bytes.getvalue()
  st.image(image, caption="Uploaded Image")

if "memory" not in st.session_state:
    st.session_state["memory"]=[]

chat=model.start_chat(history=st.session_state["memory"])
for msg in chat.history:
       st.chat_message(msg.role).write(msg.parts[0].text)
if user_prompt:
    st.chat_message("user").write(user_prompt)
    if uploaded_file is not None:
      response = chat.send_message([user_prompt,image]) 
      response.resolve()
    else:
      response = chat.send_message(user_prompt) 
    st.session_state["memory"]=chat.history  
    st.write(response.text)

