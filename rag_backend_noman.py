#1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
#from langchain_community.embeddings import BedrockEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
#from langchain.llms.bedrock import Bedrock
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock
import boto3
from io import BytesIO
import pdfplumber
from langchain_community.document_loaders import TextLoader

from langchain.schema import Document

#5c. Wrap within a function


def download_pdfs_from_s3(bucket_name, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    pdf_files = []

    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.pdf'):
            pdf_obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
            pdf_files.append(BytesIO(pdf_obj['Body'].read()))

    return pdf_files

def load_pdfs(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            documents.append(text)
    return documents

from langchain.schema import Document

def hr_index():
    # Replace 'your-bucket-name' and 'your-prefix/' with your S3 bucket name and prefix
    pdf_files = download_pdfs_from_s3('nomanragbucket', '')
    
    # Extract text from PDFs
    documents = load_pdfs(pdf_files)
    
    # Convert extracted text into Document objects
    text_documents = [Document(page_content=doc, metadata={"source": f"doc_{i}"}) for i, doc in enumerate(documents)]
    
    # Split the Text based on Character, Tokens etc.
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)
    split_documents = [data_split.split_text(doc.page_content) for doc in text_documents]
    
    # Flatten the list of split documents and re-wrap as Document objects
    flattened_documents = [Document(page_content=chunk, metadata={"source": f"doc_{i}"}) 
                           for i, doc_chunks in enumerate(split_documents) 
                           for chunk in doc_chunks]
    
    # Create Embeddings -- Client connection
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='amazon.titan-embed-text-v2:0'
    )
    
    # Create FAISS Vector Store from the embeddings
    vector_store = FAISS.from_documents(flattened_documents, embedding=data_embeddings)
    
    return vector_store


#6a. Write a function to connect to Bedrock Foundation Model - Claude Foundation Model
def hr_llm():
    llm=ChatBedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-3-haiku-20240307-v1:0',
        #model_id='anthropic.claude-v2',
        model_kwargs={
        "max_tokens":3000,
        "temperature": 0.1,
        "top_p": 0.9})
    return llm
#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def hr_rag_response(index, question):
    rag_llm = hr_llm()
    
    # Perform a similarity search using the FAISS index
    relevant_documents = index.similarity_search(question)
    
    # Combine the contents of the documents into a single string
    combined_text = "\n\n".join([doc.page_content for doc in relevant_documents])
    
    # Format the input as a list of messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides answers based on HR policies."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": combined_text}
    ]
    
    # Generate a response using the formatted messages
    response = rag_llm.invoke(messages)
    print(response)
    
    # Extract and return the text from the AIMessage object
    return response.content


# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html

import boto3




def main():
    
    # Step 1: Create the vector index from S3 PDFs
    print("Creating vector index from PDFs in S3...")
    vector_index = hr_index()
    print("Vector index created successfully.")
    
    # Step 2: Test the language model by passing a sample question
    sample_question = "When is assessment 1 due?"
    print(f"Testing with sample question: {sample_question}")
    response = hr_rag_response(vector_index, sample_question)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()