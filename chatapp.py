import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter  ##LLM library
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings #google embedding ai
import google.generativeai as genai #get google gemini api key
from langchain_community.vectorstores import FAISS #facebook ai similarty search and clustering
from langchain_google_genai import ChatGoogleGenerativeAI #
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np


load_dotenv()
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    try:
        text=""
        for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf)
            for page in pdf_reader.pages:
                text+= page.extract_text()
        return  text
    except Exception as e:
        print(e)


def process_paddle_ocr(uploaded_file, lang='en'):
    try:
        image = Image.open(uploaded_file)  # Open the image file
        image_array = np.array(image)
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_mp=True, show_log=False)
        result = ocr.ocr(image_array, cls=True)
        text = [line[1][0] for line in result[0]]
        return text
    except Exception as e:
        print(f"Error while processing PaddleOCR image: {e}")
        return None
    

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(e)


def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        print(e)

def get_conversational_chain():
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer, answer whatever asks\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.3)

        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain
    except Exception as e:
        print(e)

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
 
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        st.write("Answer : ", response["output_text"])

    except Exception as e:
        print(e)


def main():
    try:
        st.set_page_config("SummarAI", page_icon=":scroll:")  # Document name
        # st.header("Welcome to SummarAI 🚀 📚")
        st.subheader("Fast and intelligent file summarization !")


        # Sidebar for file upload
        with st.sidebar:
            st.image("img/summary_AI.png")
            st.write("---")

            st.title("📁 File's Section")
            pdf_docs = st.file_uploader(
                "Upload your PDF or Image Files", accept_multiple_files=True
            )
            

            # Automate processing when files are uploaded
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = ""
                    for file in pdf_docs:
                        if file.name.endswith("pdf"):
                            raw_text += get_pdf_text([file])  # Get the PDF text
                        elif file.name.endswith(("jpeg", "png", "jpg")):
                            raw_text += " ".join(process_paddle_ocr(file))  # Process image
                        else:
                            st.error(f"Unsupported file type: {file.name}")

                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)  # Get the text chunks
                        get_vector_store(text_chunks)  # Create vector store
                        st.success("Processing Complete!")
                    else:
                        st.error("No valid files were uploaded.")
            st.write("---")

            with st.expander("Notes (click to expand):"):
                st.markdown(
                """
                **Notes:**
                - Upload PDF or image files (JPEG, PNG, JPG).
                - Ensure the files are clear and readable for accurate processing.
                - Use the input box below to ask specific questions about the uploaded files.
                - Press "Enter" to process your question.
                - Upload Images with English text for better results.
                """
            )

            
        user_question = st.text_input("Ask a Question from your files .. ✍️📄")
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        st.button("Execute", key="execute")
                
        if user_question:
            user_input(user_question)  # Process user input
        else:
            st.info("Please enter a question to get started!")

        # Footer
        st.markdown(
            """
            <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #87A96B; padding: 15px; text-align: center;">
            <a href="https://github.com/Akilarasan1" target="_blank"> Akil </a> | Made with ❤️
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
