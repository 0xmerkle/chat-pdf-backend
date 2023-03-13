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
from custom_llm import ChatVectorDBWithPineconeMetadataFilterChain

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


def get_vectorstore_with_filter(filter):
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


def respond_with_memory_using_metadata_filter(query, vector_store, filter):
    print("running")

    chat = ChatOpenAI(temperature=0, openai_api_key=open_ai_key)

    global chat_history
    system_template = """
        You are a CHATBOT that will answer questions from the HUMAN about the documents that they provided.
        Use the context that is provided when answering the questions.
        Be conversational and not robotic.
        If the user is not asking questions directly, converse with them while encouraging them to ask questions.
        Example:
        context: {context}
        question: "tell me something about the world that is not provided by the documents"
        AI: "the documents provided are not about this topic, but here is some information about it, <information>"
        question: "tell me something relevant to the documents"
        AI: "Sure, here is information, <information>"
        -------
        {context}
    """
    system_template_default = """
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    -------
    {context}
    """
    # Be conversational and not robotic.
    # If the user is not asking questions directly, converse with them while encouraging them to ask questions
    messages = [
        SystemMessagePromptTemplate.from_template(system_template_default),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    print("prompt", prompt)
    try:
        qa = ChatVectorDBWithPineconeMetadataFilterChain.from_llm(
            llm=chat,
            vectorstore=vector_store,
            chain_type="stuff",
            qa_prompt=prompt,
            return_source_documents=True,
        )
        print("qa")
        result = qa({"question": query, "chat_history": chat_history, "filter": filter})
        reply = result["answer"]
        chat_history.append((query, reply))
        print(result["source_documents"])
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


def agent_chat_with_vectordb_qa(query, user_id):
    filter = {"user_id": {"$eq": user_id}}
    print("filter", filter)
    vector_store = get_vectore_store()
    print("vector_store", vector_store)
    r = respond_with_memory_using_metadata_filter(query, vector_store, filter=filter)
    print("r", r)
    return r
