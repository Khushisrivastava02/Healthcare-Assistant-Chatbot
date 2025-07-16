import streamlit as st
from transformers import pipeline
import nltk

# Download required nltk data
nltk.download("punkt")
nltk.download("stopwords")

# Load GPT-2 model using the pipeline
chatbot = pipeline("text-generation", model="distilgpt2")

# Set up session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def healthcare_chatbot(user_input):
    user_input_lower = user_input.lower()

    # Keyword-based quick responses
    if any(word in user_input_lower for word in ["emergency", "urgent", "help"]):
        return "⚠️ Please contact emergency services immediately."
    if "symptom" in user_input_lower:
        return "🩺 It's best to consult a healthcare provider about symptoms."
    if "appointment" in user_input_lower:
        return "📅 I can help guide you to schedule an appointment."
    if "medication" in user_input_lower:
        return "💊 Please follow your doctor’s prescription for medications."

    # Build a formatted prompt with the last 3 interactions
    past = st.session_state.chat_history[-6:]  # last 3 exchanges
    context = ""
    for i, msg in enumerate(past):
        speaker = "User" if i % 2 == 0 else "Assistant"
        context += f"{speaker}: {msg.strip()}\n"

    # Add current user input
    context += f"User: {user_input}\nAssistant:"

    # Generate response
    try:
        result = chatbot(
            context,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            truncation=True,
            num_return_sequences=1
        )

        generated_text = result[0]["generated_text"]

        # Extract only the assistant's response after the prompt
        reply = generated_text.replace(context, "").strip()

        # Remove any following "User:" or repeated prompt
        reply = reply.split("User:")[0].strip()

        # If the model fails to generate a usable reply
        if not reply or len(reply) < 5:
            reply = "🤖 I'm not sure how to answer that. Please consult a doctor."

    except Exception as e:
        reply = f"⚠️ Sorry, there was an error: {e}"

    return reply

# Streamlit UI
st.set_page_config(page_title="Healthcare Chatbot", layout="centered")
st.title("🩺 Healthcare Assistant Chatbot")
st.markdown("Ask health-related questions like symptoms, medications, or appointments.")

# Input from user
user_input = st.text_input("How can I assist you today?", "")

if st.button("Submit") and user_input:
    st.session_state.chat_history.append(user_input)
    bot_reply = healthcare_chatbot(user_input)
    st.session_state.chat_history.append(bot_reply)

# Show chat history
if st.session_state.chat_history:
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Healthcare Assistant:** {msg}")
