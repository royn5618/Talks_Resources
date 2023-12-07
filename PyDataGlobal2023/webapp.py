import streamlit as st
from config import *
from llm import ChatBot


st.image(LOGO, width=70)
st.title("💬 Hey There!")

chatty = ChatBot()

# Start
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Howdy?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = chatty.bot_chat(prompt)
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])