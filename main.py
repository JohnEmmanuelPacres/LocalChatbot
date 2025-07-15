from langchain_community.vectorstores import FAISS
# Corrected import for HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# vector storage
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts([""], embeddings)

# langchain components
template = """
Answer the question based on the context provided.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="gemma3n:e2b")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handleConversation():
    context = ""
    print("Welcome! I'm your AI assistant. Type '0' to exit.")
    while True:
        user_input = input("You: ")
        if user_input == "0":
            print("Exiting the conversation. Goodbye!")
            break
        
        retrieved_docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        result = chain.invoke({"context": context, "question":user_input}) 
        print("Pipo: ", result)

        conversation_turn = f"User: {user_input}\nAI: {result}"
        new_doc = Document(page_content=conversation_turn)
        vectorstore.add_documents([new_doc])
    

if __name__ == "__main__":
    handleConversation()