from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings
import chromadb
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
import os

app = Flask(__name__, static_folder=".", template_folder=".")
CORS(app)

# Load file
loader = UnstructuredLoader("./document/document1.pdf")
document = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = text_splitter.split_text(document[0].page_content)

# Embed chunks
ef = embedding_functions.DefaultEmbeddingFunction()
text_embeddings = ef(chunks)

class DefChromaEF(Embeddings):
    def __init__(self, ef):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]

# Store in the ChromaDB
client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="collection")

collection.upsert(
    documents=chunks,
    embeddings=text_embeddings,
    ids=["id" + str(i) for i in range(len(chunks))]
)

db = Chroma(client=client, collection_name="collection", embedding_function=DefChromaEF(ef), persist_directory="./chromadb")

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# Create LLM
llm = OllamaLLM(model="llama3.1", num_gpu=1, device="cuda")

# Prompt to generate search query for retriever
prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history", n_messages=10),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

# Retriever Chain: retrieve documents from vector store relevant to user query and chat history
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

# Prompt to get response from LLM based on chat history
prompt_get_answer = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\\n\\n{context}"),
    MessagesPlaceholder(variable_name="chat_history", n_messages=10),
    ("user", "{input}"),
])

# Document Chain: send relevant documents, chat history and user query to LLM
document_chain = create_stuff_documents_chain(llm, prompt_get_answer)

# Create conversational retrieval chain: combine retriever and document chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    chat_history = data.get('chat_history', [])

    langchain_history = [
        HumanMessage(content=msg['content']) if msg['role'] == 'user' else AIMessage(content=msg['content'])
        for msg in chat_history
    ]

    recent_user_message = HumanMessage(content=user_input)

    # Invoke the chain
    response = retrieval_chain.invoke({
        "chat_history": langchain_history,
        "input": recent_user_message.content
    })

    return jsonify({"response": response['answer']})

# Serve the frontend
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True)