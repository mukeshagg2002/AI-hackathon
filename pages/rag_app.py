from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

f = open("keys\gemini_api_key.txt")
key = f.read()

st.title("LangChain RAG System 🤖📊🔮✨")
uploaded_file = st.file_uploader("Upload your context PDF:")
user_prompt=st.text_input("write your query")
button=st.button("Generate 👨‍💻")
if button==True:
    st.subheader("Question")
    st.title(user_prompt)
output_parser = StrOutputParser()

# intitalize the model 

chat_model = ChatGoogleGenerativeAI(google_api_key=key, 
                                   model="gemini-1.5-pro-latest")

#  intitalize the embeddings model 
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyC2Bztff9XtDCDrCJfMJ8py9JaT8VkwSlY", 
                                               model="models/embedding-001")

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

# loading the pdf document
if uploaded_file:
    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load_and_split()

    # Split the documents in chunk
    text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)

    chunks = text_splitter.split_documents(pages)

    # Store the chunks in vector store
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
    db.persist()

    # Setting a Connection with the ChromaDB
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

    # Converting CHROMA db_connection to Retriever Object
    retriever = db_connection.as_retriever(search_kwargs={"k": 10})



    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
        )
if button ==True:
    response = chain.invoke(user_prompt)
    st.write(response)