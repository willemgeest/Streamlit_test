import streamlit as st
from pathlib import Path
import requests
import torch
from torchvision.models import resnet50

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

st.title('testpagina Streamlit')

@st.cache
def get_class_model_Drive():
    save_dest = Path('checkpoints')
    save_dest.mkdir(exist_ok=True)
    f_checkpoint = Path("checkpoints/resnet50-19c8e357.pth")
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take a while! \n Don't stop it!"):
            download_file_from_google_drive('1BhJaGO6ENvk5va8zVaSJsl8XFCVckCu6', f_checkpoint)
            st.text(f_checkpoint)
    
    model = resnet50(pretrained=True)
    return model
  
model =  get_class_model_Drive()

st.text(model)
