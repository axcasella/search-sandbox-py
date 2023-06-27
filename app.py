import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle
import os

# Sidebar contents
with st.sidebar:
    st.title('Demo of LLM semantic search')
    add_vertical_space(3)
    st.markdown('''
    ### Made by Alex Casella
    ''')

# Main page contents
def main():
    load_dotenv()

    st.header("Simple search of 10-Ks and 10-Qs")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Get rid of .pdf from name
        store_file_name = pdf.name[:-4]

        if os.path.exists(f"{store_file_name}.pk1"):
            # read the file
            with open(f"{store_file_name}.pk1", "rb") as f:
                vectorStore = pickle.load(f)

            st.write('File already exists, loaded embeddings from disk')
        else:
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Use langchain to split into chunks
            splitter = RecursiveCharacterTextSplitter(["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text=text)
            # st.write("Chunk:")
            st.write("File broken into chunks: ", len(chunks))
            # st.write(chunks)

            # Embed chunks
            embeddings = OpenAIEmbeddings()

            # Create vector store using Meta's FAISS store
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            num_vectors = vectorStore.index.ntotal
            st.write(f"New file, created vector store with {num_vectors} vectors")

            with open(f"{store_file_name}.pk1", "wb") as f:
                pickle.dump(vectorStore, f)
            
        # Take in search query
        query = st.text_input("Search for:")
        if query:
            # Get top results
            k = 5
            docs = vectorStore.similarity_search(query=query, k=k)

            # Ask LLM to give final result
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write("Answer:")
            st.write(response)

            st.write(f"Top {k} results:")
            cleaned_docs = [doc.page_content.replace('\n', ' ') for doc in docs]
            st.write(cleaned_docs)


if __name__ == "__main__":
    main()