import streamlit as st
st.title('testpagina Streamlit')

@st.cache
def get_class_model_Drive():
    save_dest = Path('checkpoints')
    save_dest.mkdir(exist_ok=True)
    f_checkpoint = Path(torch.hub.get_dir() + "/checkpoints/resnet50-19c8e357.pth")
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            download_file_from_google_drive('1BhJaGO6ENvk5va8zVaSJsl8XFCVckCu6', f_checkpoint)

    model = resnet50(pretrained=True)
    return model
  
model =  get_class_model_Drive()

st.text(model)
