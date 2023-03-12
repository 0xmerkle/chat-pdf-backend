import firebase_admin
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import os
import tempfile
from langchain.document_loaders import PagedPDFSplitter
from firebase_admin import auth, credentials

creds_path = os.path.join(os.path.dirname(__file__), "google-credentials.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


from google.cloud import storage
from pinecone_utils import (
    load_pages_to_pinecone,
    get_embeddings,
)
from query_utils import agent_chat_with_vectordb_qa

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {"origins": "http://localhost:3000", "supports_credentials": True}
    },
)
app.secret_key = os.environ.get("SECRET_KEY")

# Initialize the Firebase Admin SDK
key_path = os.path.join(os.path.dirname(__file__), "service_account_key.json")
cred = credentials.Certificate(key_path)
firebase_admin.initialize_app(cred)


def load_pdf(filePath):
    loader = PagedPDFSplitter(filePath)
    pages = loader.load_and_split()
    return pages


def process_pdf(file):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        file.save(tmp_file.name)
        tmp_file_path = tmp_file.name
    pages = load_pdf(tmp_file_path)
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
        token = auth_header.split(" ")[1]
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


project_id = "chat-pdf"
bucket_name = "chat-pdf-text"


@app.route("/upload", methods=["POST"])
async def upload():
    try:
        file = request.files["upload"]
        if file:
            client = storage.Client(project=project_id)
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(file.filename)
            content = file.read()
            blob.upload_from_string(content, content_type=file.content_type)
            pdf_uri = f"gs://{bucket_name}/{file.filename}"
            print(pdf_uri)
            ## works
            gcs_destination_uri = "gs://chat-pdf-text/results/"
            docs = async_detect_document(pdf_uri, gcs_destination_uri)
            embeddings = get_embeddings(docs)

            print("loading to pinecone...")
            load_pages_to_pinecone(docs, embeddings)
            print("annotated text", docs)

            return jsonify(text=docs, status=200)
        else:
            return "No file found"
    except Exception as e:
        print(e)
        return "Error"


@app.route("/loadPdf", methods=["POST"])
async def loadPdf():
    if not request.user:
        abort(401, description="Authorization header not found")
    try:
        file = request.files.get("upload")
        print(file)
        if file:
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
            res = agent_chat_with_vectordb_qa(query)
            return jsonify(res=res, status=200)
        else:
            return "No file found"
    except Exception as e:
        print(e)
        return "Error"


if __name__ == "__main__":
    app.run(port=8080, debug=True)
