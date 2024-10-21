import streamlit as st
import wikipediaapi
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  # for extracting images
from PIL import Image

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
image_metadata = []

# Function to get VLSI data from Wikipedia
def get_wikipedia_data(topic="VLSI"):
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(topic)
    if not page.exists():
        return ""
    return page.text

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print(chunks)
    return chunks

# Function to create FAISS index from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to build the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context", don't provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Function to extract images from uploaded PDFs
def extract_images_from_pdf(pdf_docs):
    global image_metadata
    images_list = []

    for pdf in pdf_docs:
        pdf_file = fitz.open(stream=pdf.read(), filetype="pdf")
        page_nums = len(pdf_file)

        for page_num in range(page_nums):
            page_content = pdf_file[page_num]
            images_list.extend(page_content.get_images(full=True))

        if len(images_list) == 0:
            st.warning(f"No images found in {pdf.name}")
        else:
            for i, image in enumerate(images_list, start=1):
                xref = image[0]
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"image_{pdf.name.split('.')[0]}_{i}.{image_ext}"

                image_path = os.path.join('images', image_name)
                os.makedirs('images', exist_ok=True)
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)

                default_title = f"Image from {pdf.name} - page {page_num + 1}"
                image_title = st.text_input(f"Title for Image {i} from {pdf.name}", default_title)

                image_metadata.append({
                    "image_path": image_path,
                    "page_num": page_num,
                    "title": image_title,
                    "description": f"{image_title} (Page {page_num + 1})"
                })

                st.image(image_path, caption=image_title)
                st.success(f"Saved {image_name} with title: {image_title}")

# Function to handle user input and provide a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)

    relevant_images = [img for img in image_metadata if user_question.lower() in img['description'].lower()]
    for img in relevant_images:
        st.image(img["image_path"], caption=img["description"])

    st.write("Reply:", response["output_text"])

# Main function for the Streamlit app
def main():
    st.set_page_config("Chat VLSI & PDF")
    st.title("Artificial Intelligence: VLSI Guide with PDF Support")

    # Ask the user for a question
    user_question = st.text_input("Ask a question related to VLSI or the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        data_source = st.radio("Choose Data Source", ("Wikipedia (VLSI)", "Upload PDFs"))

        if data_source == "Wikipedia (VLSI)":
            if st.button("Fetch VLSI Wikipedia Data"):
                with st.spinner("Fetching Wikipedia data..."):
                    vlsi_text = get_wikipedia_data("VLSI")
                    text_chunks = get_text_chunks(vlsi_text)
                    get_vector_store(text_chunks)
                    st.success("Wikipedia VLSI data processed and stored!")
        
        elif data_source == "Upload PDFs":
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
            if st.button("Submit & Process PDFs"):
                with st.spinner("Processing..."):
                    extract_images_from_pdf(pdf_docs)
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF files processed and stored!")

if __name__ == "__main__":
    main()
