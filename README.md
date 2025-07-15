# LocalChatbot
LLM used: Gemma 3n:e2b

Simple **Windows** chatbot in **Python** for understanding local LLM usage and try implementing semantic persistency using vector storage.

# Documentation
### Dependencies
```
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
```
Used langchain to connect the Ollama for LLM data retrieval, and use FAISS which is an open source library by Meta for efficient similar searches, used for context persistency. Used huggingface for multiple ML framework integration and prototyping.

This approach allows the program to remember past interactions (though not connected between sessions due to lack of context memory persistency) through vector storage. It supports relevant past exchanges even if keywords don't exactly match and provides the most relevant parts of history to the LLM. Through **FAISS** it allows fast similarity search as the conversation grows.

## Vector storage

### Initialization
Creates an empty FAISS index initialized with empty string, then prepares to store later add conversation turns.

```
vectorstore = FAISS.from_texts([""], embeddings)
```

### During Conversation
Retrieves k most similar past conversation turns; uses the embeddings to find semantically related past interactions.

```
retrieved_docs = vectorstore.similarity_search(user_input, k=3) 
```

### Context Building
Combines the k relevant past interactions to provide context to LLM

```
context = "\n".join([doc.page_content for doc in retrieved_docs])
```

### Storing new interactions
After each exchange, stores the full user-AI interaction as a new document, which builds a growing memory of the conversation

```
conversation_turn = f"User: {user_input}\nAI: {result}"
new_doc = Document(page_content=conversation_turn)
vectorstore.add_documents([new_doc])
```

# Usage
## Use this command in terminal if ever not inside the python environment yet
`python -m venv chatbot`

If not activated yet, try using this in terminal:

`source chatbot/Scripts/activate`

When there is a '(chatbot)' beside user address, this means user already inside the environment.

## For running the program type this in terminal
`python main.py`

It may take time to run depending on hardware and LLM constraints.

# Future enhancement
- Preload initial knowledge
- Add memory persistency to save conversation between sessions
- Try different embedding models and chunking strategies

# Reference
- [Create a LOCAL Python AI Chatbot In Minutes Using Ollama](https://www.youtube.com/watch?v=d0o89z134CQ&t=274s)
- [HuggingFaceEmbeddings](https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html)
- [faiss](https://github.com/facebookresearch/faiss)

