import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sidebar contents
with st.sidebar:
    st.title('Test PE Search App')
    st.markdown('''
    ## Simple search of 10-Ks and 10-Qs
    ''')
    add_vertical_space(5)
    st.write('Made by Alex Casella')

# Main page contents
def main():
    st.header("Search input")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Use langchain to split into chunks
        splitter = RecursiveCharacterTextSplitter(separators=[".", ",", "\n"], chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text=text)

        st.write(chunks)

if __name__ == "__main__":
    main()