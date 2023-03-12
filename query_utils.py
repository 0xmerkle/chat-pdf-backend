from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pinecone
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain.chains import ChatVectorDBChain

load_dotenv()

import os

open_ai_key = os.environ.get("OPENAI_API_KEY")
pinecone_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("INDEX_NAME")
chat_history = []
limit = 3750


def get_vectorstore():
    pinecone.init(api_key=pinecone_key, environment="us-west1-gcp")
    index = pinecone.Index(index_name)

    embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)
    vectorstore = Pinecone(index, embeddings.embed_query, "text")

    print("vectorstore", vectorstore)
    return vectorstore


def respond_with_memory(query, vector_store):
    print("running")

    chat = ChatOpenAI(temperature=0, openai_api_key=open_ai_key)
    global chat_history
    system_template = """
        You are a CHATBOT that will answer questions from the HUMAN about the documents that they provided.
        Use the context that is provided when answering the questions.
        Be conversational and not robotic.
        If the user is not asking questions directly, converse with them while encouraging them to ask questions.
        -------
        {context}
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    print("prompt", prompt)
    try:
        qa = ChatVectorDBChain.from_llm(
            llm=chat, vectorstore=vector_store, qa_prompt=prompt
        )
        print("qa")
        result = qa({"question": query, "chat_history": chat_history})
        reply = result["answer"]
        chat_history.append((query, reply))
        # if chat_history.__len__() > 10:
        #     chat_history = chat_history[-10:]
        # print("result", result)
        print("chat_history", chat_history)
        return reply
    except Exception as e:
        print("e", e)
        return "I don't know"


def get_vectore_store():
    embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)

    pinecone.init(api_key=pinecone_key, environment="us-west1-gcp")

    vector_store = Pinecone.from_existing_index(index_name, embeddings, "text")
    return vector_store


def agent_chat_with_vectordb_qa(query):
    vector_store = get_vectore_store()
    print("vector_store", vector_store)
    r = respond_with_memory(query, vector_store)
    print("r", r)
    return r
