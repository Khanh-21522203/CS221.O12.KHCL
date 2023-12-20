import streamlit as st
from streamlit_option_menu import option_menu
from model import VDCNN
import torch
from utils import transfrom_input_AgNews

with st.sidebar:
    selected = option_menu(
                menu_title="VDCNN",
                options=["AG News", "UIT student feedback"],  # required
                menu_icon="cast",
                default_index=0,
                orientation="vertical",
            )
st.title(":violet[Very Deep Convolutional Networks for Text Classification]")
if selected == "AG News":
    if 'agNews' not in st.session_state:
        model = VDCNN(n_classes=4, depth=9)
        checkpoint = torch.load('./trained_models/model_ag_9.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        classNames = ["World", "Sports", "Business", "ScienceAndTech"]
        st.session_state.agNews = {"model": model, "classNames": classNames}
    model = st.session_state.agNews["model"]
    classNames = st.session_state.agNews["classNames"]

if selected == "UIT student feedback":
    if 'UITfeedback' not in st.session_state:
        model = VDCNN(n_classes=3, depth = 49, num_embedding=1024, shortcut=True, maxpooling=True)
        checkpoint = torch.load('./trained_models/model_Vietnamese.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        classNames = ["negative", "neutral", "positive"]
        st.session_state.UITfeedback = {"model": model, "classNames": classNames}
    model = st.session_state.UITfeedback["model"]
    classNames = st.session_state.UITfeedback["classNames"]

input_text = st.text_area("")
button = st.button("Go")

if input_text and button:
    encoded_input = transfrom_input_AgNews(input_text)
    model.eval()
    with torch.no_grad():
        st.text(encoded_input)
        predictions = model(encoded_input)
        indices = torch.argmax(predictions.cpu(), dim=1).item()
        st.text(f"Prediction: {classNames[indices]}")

