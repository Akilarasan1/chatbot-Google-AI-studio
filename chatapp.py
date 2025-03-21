import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter  ##LLM library
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings #google embedding ai
import google.generativeai as genai #get google gemini api key
from langchain.vectorstores import FAISS #facebook ai similarty search and clustering
from langchain_google_genai import ChatGoogleGenerativeAI #
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from paddleocr import PaddleOCR

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


def process_paddle_ocr(image_path, lang ='en'):
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang,use_mp=True,show_log = False)  # Initialize OCR model
        result = ocr.ocr(image_path, cls=True)
        text = [line[1][0] for line in result[0]]
        return text
    
    except Exception as e:
        print(f'while Error Processing paddle_ocr image processing:::: {e}')
        pass

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(e)


def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-004")
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

        print(response)
        st.write("Reply: ", response["output_text"])

    except Exception as e:
        print(e)



def main():
    try:
        st.set_page_config("Image and PDF Summarization", page_icon = ":scroll:") # document name
        st.header("Image and PDf extract Summarization 📚")

        user_question = st.text_input("Ask a Question from the PDF and Images Files uploaded .. ✍️📝")

        if user_question:
            user_input(user_question)

        with st.sidebar:

            st.image("img/Robot.jpg")
            st.write("---")
            
            st.title("📁 PDF File's Section")
            pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."): # user friendly message.
                    if pdf_docs.endswith('pdf'):
                        raw_text = get_pdf_text(pdf_docs) # get the pdf text
                    elif pdf_docs.endswith(['jpeg', 'png', 'jpg']):
                        raw_text = process_paddle_ocr(pdf_docs)
                    else:
                        print("Please Upload relevant PDF or Images")

                    text_chunks = get_text_chunks(raw_text) # get the text chunks
                    get_vector_store(text_chunks) # create vector store
                    st.success("Done")
            
            st.write("---")
            st.image("img/gkj.jpg")
            st.write("AI App created by @ Akil")  # add this line to display the image


        st.markdown(
            """
            <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
                © <a href="https://github.com/Akilarasan1" target="_blank"> Akilarasan </a> | Made with ❤️
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()