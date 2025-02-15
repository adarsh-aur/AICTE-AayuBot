import streamlit as st
from transformers import pipeline, BertForQuestionAnswering, BertTokenizer
import nltk

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the pre-trained SQuAD model and tokenizer directly from Hugging Face
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # Pre-trained model from Hugging Face
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Initialize BioBERT for question answering using the pre-trained SQuAD model
bio_qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Initialize GPT-3 (or DialoGPT) for text generation (for general chatbot conversation)
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Function to process healthcare-related questions
def healthcare_chatbot(user_input):
    user_input = user_input.lower()
    
    # Define common responses based on keywords
    responses = {
        "symptom": "I recommend consulting a doctor for an accurate diagnosis.",
        "appointment": "Would you like to schedule an appointment with a healthcare professional?",
        "medication": "Make sure to follow your doctor’s prescription. If you have concerns, consult a medical expert.",
        "emergency": "In case of an emergency, please contact the nearest hospital or dial emergency services immediately.",
        "fever": "For fever, stay hydrated and rest. If it persists, consult a doctor.",
        "headache": "Headaches can be caused by stress or dehydration. Drink water and take rest. If severe, seek medical advice."
    }
    
    # Return pre-defined response based on keyword match
    for keyword, response in responses.items():
        if keyword in user_input:
            return response

    # Use BioBERT for healthcare-related questions using the pre-trained SQuAD model
    context = "Hugging Face is creating a tool that democratizes AI in healthcare and many other fields."  # Sample context
    answer = bio_qa_pipeline(question=user_input, context=context)

    if answer['score'] > 0.5:  # Only use the answer if the confidence score is high
        return answer['answer']
    else:
        # Fallback to GPT-2 (or DialogGPT) for general conversation if BioBERT can't answer confidently
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']

# Streamlit App
def main():
    st.set_page_config(page_title="AayuBot", page_icon="💊", layout="centered")
    
    # Enhanced Dynamic CSS styling
    st.markdown(
        f"""
        <style>
        body {{
            font-family: 'Arial', sans-serif;
            background: linear-gradient(45deg, #ff7e5f, #feb47b);
            animation: backgroundShift 10s ease-in-out infinite;
            margin: 0;
            padding: 0;
        }}

        @keyframes backgroundShift {{
            0% {{background: linear-gradient(45deg, #ff7e5f, #feb47b);}}
            25% {{background: linear-gradient(45deg, #feb47b, #ff7e5f);}}
            50% {{background: linear-gradient(45deg, #ff7e5f, #feb47b);}}
            75% {{background: linear-gradient(45deg, #feb47b, #ff7e5f);}}
            100% {{background: linear-gradient(45deg, #ff7e5f, #feb47b);}}
        }}

        .main-container {{
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 20px;
            margin-top: 30px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease;
        }}

        h1 {{
            font-size: 2.5em;
            color: #ffffff;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }}

        .stTextInput {{
            padding: 10px;
            font-size: 16px;
            border-radius: 10px;
            border: 1px solid #ff7e5f;
            width: 100%;
            margin-bottom: 20px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }}

        .stTextInput:hover {{
            border-color: #feb47b;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
        }}

        .stButton {{
            background-color: #ff7e5f;
            color: white;
            font-size: 16px;
            padding: 12px;
            width: 100%;
            border-radius: 10px;
            cursor: pointer;
            border: none;
            transition: background-color 0.3s ease;
        }}

        .stButton:hover {{
            background-color: #feb47b;
        }}

        .response-box {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
            color: #333;
            transition: transform 0.3s ease;
        }}

        .response-box:hover {{
            transform: scale(1.05);
        }}

        .response-title {{
            font-size: 1.2em;
            font-weight: bold;
        }}

        .disclaimer {{
            font-size: 0.9em;
            color: #aaa;
            text-align: center;
            margin-top: 30px;
        }}

        .stAlert {{
            background-color: #ffcccc;
            color: #ff0000;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }}

        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown("<h1>AayuBot</h1>", unsafe_allow_html=True)
    st.write("Welcome to your AI-powered healthcare assistant! Ask me any health-related questions.")
    
    user_input = st.text_input("How can I assist you today?", placeholder="Type your question here...")

    if st.button("Ask Now"):
        if user_input.strip():
            st.markdown(f"**👤 You:** {user_input}")
            with st.spinner("Processing your query..."):
                response = healthcare_chatbot(user_input)
            st.success("Response Generated!")
            st.markdown(f"<div class='response-box'><div class='response-title'>💬 Chatbot:</div>{response}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='stAlert'>Please enter a valid question.</div>", unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='disclaimer'>⚠️ Disclaimer: This chatbot provides general health advice. Always consult a medical professional for serious concerns.</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
