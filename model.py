import streamlit as st
import torch
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification
import numpy as np

# تحميل النموذج والتوكنايزر
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        "./saved_model",
        num_labels=2,
        id2label={0: 'benign', 1: 'malicious', 2: 'unknown'},
        label2id={'benign': 0, 'malicious': 1, 'unknown': 2}
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def predict(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_label = np.argmax(probabilities)
    
    return {
        'label': model.config.id2label[predicted_label],
        'probability_benign': float(probabilities[0]),
        'probability_malicious': float(probabilities[1])
    }

st.title("تصنيف النصوص: كويس أو مش كويس")
st.write("أدخلي النص وسيتم تصنيفه إلى 'benign' (كويس) أو 'malicious' (مش كويس)")

user_input = st.text_area("أدخلي النص هنا:", height=200)

if st.button("صنّف النص"):
    if user_input.strip() == "":
        st.error("من فضلكِ أدخلي نصًا للتصنيف!")
    else:
        result = predict(user_input)
        st.success(f"**النتيجة:** النص هو **{result['label']}**")
        st.write(f"**احتمالية benign (كويس):** {result['probability_benign']:.2%}")
        st.write(f"**احتمالية malicious (مش كويس):** {result['probability_malicious']:.2%}")