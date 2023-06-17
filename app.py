import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv


load_dotenv()
AUTHORIZATION_TOKEN = os.getenv("AUTHORIZATION_TOKEN")

st.title("🤗HuggingFace Pipeline Implementation")

with st.expander("NSFW words classification"):
    st.title("NSFW words classification")
    text_input = st.text_input("Enter sentence here")
    API_URL = "https://api-inference.huggingface.co/models/michellejieli/NSFW_text_classifier"
    headers = {"Authorization": f"Bearer {AUTHORIZATION_TOKEN}"}


    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text_input
    })
    if text_input:
        if output[0][0]['label'] == 'SFW':
            st.progress(output[0][0]['score'])
            st.text("SFW")
            st.progress(1.000 - output[0][0]['score'])
            st.text("NSFW")
        else:
            st.progress(output[0][0]['score'])
            st.text("NSFW")
            st.progress(1.000 - output[0][0]['score'])
            st.text("SFW")

with st.expander("Image Generation"):
    API_URL = "https://api-inference.huggingface.co/models/cloudqi/cqi_text_to_image_pt_v0"
    headers = {"Authorization": f"Bearer {AUTHORIZATION_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    prompt = st.text_input("Enter Prompt")
    image_bytes = query({
        "inputs": prompt,
    })
    image = Image.open(BytesIO(image_bytes))
    if image:
        st.image(image)
        temp_file = BytesIO()
        image.save(temp_file, format='JPEG')
        temp_file.seek(0)
        st.download_button("Download Generated Image", temp_file, file_name=f"{prompt}.jpg")

with st.expander("Text Generation"):
    API_URL = "https://api-inference.huggingface.co/models/Moxis/Harry_Potter_text_generation"
    headers = {"Authorization": f"Bearer {AUTHORIZATION_TOKEN}"}


    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    text = st.text_area("Enter text", "")
    output = ""
    output = query({
        "inputs": text,
    })
    if output and isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
        gen = output[0]["generated_text"]
        text = st.text(gen)

with st.expander("Text Summarizer"):

    API_URL = "https://api-inference.huggingface.co/models/gavin124/gpt2-finetuned-cnn-summarization-v2"
    headers = {"Authorization": f"Bearer {AUTHORIZATION_TOKEN}"}


    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    text2 = st.text_area("Enter text for summarization")
    output = query({
        "inputs": text2,
    })
    if output:
        st.write(output)

with st.expander("Translate Hindi to English"):
    API_URL = "https://api-inference.huggingface.co/models/snehalyelmati/mt5-hindi-to-english"
    headers = {"Authorization": f"Bearer {AUTHORIZATION_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    text3 = st.text_area("Enter text to translate to english")
    output = query({
        "inputs": text3,
    })

    if text3:
        st.write(output[0]['generated_text'])

