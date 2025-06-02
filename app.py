import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Custom CSS for light blue background ---
st.markdown(
    """
    <style>
    body {
        background-color: #e6f2ff;
    }
    .stApp {
        background-color: #e6f2ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = load_model("Model.h5")
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Suggestions for each class
suggestions = {
    'Glioma': (
        "Glioma is a type of tumor that occurs in the brain and spinal cord. "
        "These tumors are often malignant and can grow rapidly, affecting critical brain functions. "
        "Treatment typically includes a combination of surgery, radiation therapy, and chemotherapy. "
        "Early diagnosis and intervention are crucial for improving outcomes. Please consult a neurologist or neuro-oncologist as soon as possible for detailed diagnosis and treatment planning."
    ),
    'Meningioma': (
        "Meningiomas are tumors that arise from the meninges, the protective layers of the brain and spinal cord. "
        "Most meningiomas are benign and slow-growing, but they can cause problems depending on their size and location. "
        "Treatment options may include regular monitoring, surgical removal, or radiation therapy. "
        "It's important to get a detailed scan and consult with a neurologist or neurosurgeon to determine the best course of action."
    ),
    'No Tumor': (
        "Your brain MRI does not show any signs of a tumor. This is a positive result. "
        "However, if you are experiencing symptoms like headaches, vision problems, or memory issues, please consult a medical professional to explore other possible causes. "
        "Maintaining brain health through regular check-ups, hydration, sleep, and mental stimulation is always recommended."
    ),
    'Pituitary': (
        "Pituitary tumors originate in the pituitary gland and can affect hormone production, which may lead to symptoms like vision changes, fatigue, mood swings, and hormonal imbalances. "
        "Most pituitary tumors are benign and slow-growing. Treatment may involve medication, hormone therapy, surgery, or radiation depending on the tumor type and effects. "
        "An endocrinologist and a neurosurgeon should be consulted for proper hormone testing and imaging to plan an effective treatment strategy."
    )
}


# Streamlit UI
st.title("🧠 Brain Tumor Diagnostic Chatbot")
st.write("Upload an MRI brain scan to get a diagnosis and medical advice from DoctorBot 🤖")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption="🖼️ MRI Scan", use_column_width=True)

    if st.button("🧪 Diagnose"):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.markdown("### 🤖 DoctorBot's Diagnosis")
        st.success(f"🧠 Predicted Tumor Type: **{predicted_class}**")
        st.info(f"💬 Suggestion: {suggestions[predicted_class]}")
