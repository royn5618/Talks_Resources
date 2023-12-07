import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

from config import *


class ChatBot:
    def __init__(self):
        self.llm = ChatOpenAI(openai_api_key=os.environ[OPENAI_API_KEY],
                              model_name=GPT_MODEL,
                              temperature=0)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.environ[OPENAI_API_KEY],
                                           model=EMBEDDING_MODEL,
                                           chunk_size=16)
        self.vector_db = Chroma(persist_directory=VECDB_DIR,
                                embedding_function=self.embeddings)
        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                               return_messages=True)
        self.retriever = self.vector_db.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(self.llm,
                                                        retriever=self.retriever,
                                                        memory=self.memory)

    def bot_chat(self, question):
        response = {}
        result = self.qa({"question": question})
        response["role"] = "assistant"
        response['content'] = result['answer']
        return response
