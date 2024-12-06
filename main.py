import ollama
import chromadb
from flask import Flask
from pypdf import PdfReader
from markupsafe import escape

reader = PdfReader("leave_policy.pdf")
documents = reader.pages

client = chromadb.Client()
collection = client.create_collection(name="docs")


# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d.extract_text())
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d.extract_text()]
  )
  

app = Flask(__name__)

@app.route("/<prompt>")
def query(prompt):
    # generate an embedding for the prompt and retrieve the most relevant doc
    response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large"
    )
    results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
    )
    data = results['documents'][0][0]
    # generate a response combining the prompt and data we retrieved in step 2
    output = ollama.generate(
    model="llama2",
    prompt=f"Using this data: {data}. Respond to this prompt: {escape(prompt)}"
    )
    return output['response']