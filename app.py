import streamlit as st
from transformers import pipeline
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load GPT-2 with TensorFlow backend
chatbot = pipeline("text-generation", model="distilgpt2", framework="tf")

# Initialize a list to keep chat history for context
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def healthcare_chatbot(user_input):
    user_input_lower = user_input.lower()

    # Safety check for emergencies
    if any(word in user_input_lower for word in ["emergency", "urgent", "help"]):
        return "If this is an emergency, please call your local emergency number immediately."

    # Basic healthcare tips with keywords
    if "symptom" in user_input_lower:
        return "It's important to consult a doctor if you are experiencing symptoms."
    elif "appointment" in user_input_lower:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input_lower:
        return "Make sure to take your prescribed medications regularly. Consult your doctor if you have concerns."

    # Combine chat history + new user input for context
    context = " ".join([f"User: {msg}" if i % 2 == 0 else f"Assistant: {msg}" 
                        for i, msg in enumerate(st.session_state.chat_history[-6:])])  # last 3 exchanges
    context += f" User: {user_input}"

    # Generate a response with context, limit length for practical use
    response = chatbot(context, max_length=150, num_return_sequences=1)
    generated_text = response[0]['generated_text']

    # Remove repeated user input from generated text if any
    reply = generated_text.split(user_input)[-1].strip()
    if not reply:
        reply = generated_text  # fallback

    return reply

def main():
    st.set_page_config(page_title="Healthcare Assistant Chatbot", layout="centered")
    st.title("🩺 Healthcare Assistant Chatbot")
    st.write("Ask health-related queries like symptoms, appointments, or medication advice.")

    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit") and user_input:
        st.session_state.chat_history.append(user_input)

        response = healthcare_chatbot(user_input)
        st.session_state.chat_history.append(response)

    # Display chat history
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Healthcare Assistant:** {msg}")

if __name__ == "__main__":
    main()
