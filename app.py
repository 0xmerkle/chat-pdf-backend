import firebase_admin
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import os
import tempfile
from langchain.document_loaders import PagedPDFSplitter
from firebase_admin import auth, credentials
import json
from dotenv import load_dotenv

from pinecone_utils import (
    load_pages_to_pinecone,
    get_embeddings,
)
from query_utils import agent_chat_with_vectordb_qa

load_dotenv()

# Initialize the Firebase Admin SDK
service_account_key_json = os.environ.get("SERVICE_ACCOUNT_KEY")
service_account_key = json.loads(service_account_key_json)
cred = credentials.Certificate(service_account_key)
firebase_admin.initialize_app(cred)

origins = json.loads(os.environ.get("ORIGINS"))

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": origins, "supports_credentials": True}},
)


def add_user_id(docs, user_id):
    for doc in docs:
        doc.metadata = {**doc.metadata, "user_id": user_id}
    return docs


def load_pdf(filePath):
    loader = PagedPDFSplitter(filePath)
    pages = loader.load_and_split()
    return pages


def load_pdf_and_add_metadata(filePath):
    loader = PagedPDFSplitter(filePath)
    pages = loader.load_and_split()
    f = pages[0]
    print(f.metadata)
    print("adding metadata to pages for user", request.user.uid)
    augmented_pages = add_user_id(pages, request.user.uid)
    print("augmented_pages", augmented_pages[0])
    return augmented_pages


def process_pdf(file):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        file.save(tmp_file.name)
        tmp_file_path = tmp_file.name
    # pages = load_pdf(tmp_file_path)
    pages = load_pdf_and_add_metadata(tmp_file_path)
    os.remove(tmp_file_path)
    return pages


# Define the Firebase authentication middleware
def verify_token(token):
    try:
        # Verify the user's Firebase ID token
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token["uid"]
        # Retrieve the user's data from the Firebase Auth user record
        user = auth.get_user(uid)
        return user
    except:
        return None


@app.before_request
def authenticate_user():
    # Get the Firebase ID token from the Authorization header
    auth_header = request.headers.get("Authorization")
    print("auth_header", auth_header)
    if auth_header:
        token = auth_header.split(" ")[0]
        # Verify the user's Firebase ID token
        user = verify_token(token)
        print(user, "user")
        if user:
            # Store the authenticated user in Flask's g object
            request.user = user
    else:
        request.user = None


@app.route("/", methods=["GET"])
def hello():
    return "Hello, World!"


@app.route("/loadPdf", methods=["POST"])
async def loadPdf():
    if not request.user:
        abort(401, description="Authorization header not found")
    try:
        file = request.files.get("upload")
        print(file)
        if file:
            print(request.user)
            pages = process_pdf(file)

            print("loading to pinecone...")
            print(len(pages))
            index_name = load_pages_to_pinecone(pages=pages)

            return jsonify(index_name=index_name, status=200)
        else:
            return "No file found"
    except Exception as e:
        print(e)
        return "Error"


@app.route("/chat_with_agent", methods=["POST"])
async def chat_with_agent():
    if not request.user:
        abort(401, description="Authorization header not found")
    try:
        data = request.get_json()
        query = data["query"]
        if query:
            res = agent_chat_with_vectordb_qa(query, request.user.uid)
            return jsonify(res=res, status=200)
        else:
            return "No file found"
    except Exception as e:
        print(e)
        return "Error"


@app.route("/get_context_info_from_documents", methods=["POST"])
async def get_context_info_from_documents():
    if not request.user:
        abort(401, description="Authorization header not found")
    try:
        initial_query = "describe the context of the documents that are provided and suggest things i could ask about"
        if initial_query:
            res = agent_chat_with_vectordb_qa(initial_query, request.user.uid)
            return jsonify(res=res, status=200)
        else:
            return "No file found"
    except Exception as e:
        print(e)
        return "Error"


if __name__ == "__main__":
    app.run(port=8080, debug=True)
