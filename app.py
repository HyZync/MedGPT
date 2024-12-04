import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import time

# Load the fine-tuned BioGPT model and tokenizer
model_path = r"D:\biogpt-finetuned-model"  # Path where the model is saved
tokenizer_path = r"D:\biogpt-finetuned-tokenizer"  # Path where the tokenizer is saved

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Define a function to generate text based on a user prompt with sampling parameters
def generate_response(prompt, max_length=300, temperature=0.7, top_k=50):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output using the model with specified sampling parameters
    with torch.no_grad():  # Disable gradient calculation for evaluation
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,  # Adjust randomness
            top_k=top_k,              # Limit to top-k tokens
            do_sample=True             # Enable sampling
        )

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Truncate the output at the last complete sentence
    if "." in generated_text:
        truncated_text = generated_text[:generated_text.rfind(".") + 1]
    else:
        truncated_text = generated_text  # Return the whole text if no period found

    return truncated_text

# Streamlit UI setup
st.set_page_config(page_title="Tempus MedGPT", page_icon=":guardsman:", layout="wide")

# Add custom CSS to style the page
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4B9CD3;
        }
        .test-phase-banner {
            background-color: #f0ad4e;
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .input-box {
            font-size: 18px;
            height: 50px;
            width: 70%;
            margin-bottom: 20px;
        }
        .response-box {
            font-size: 20px;
            padding: 20px;
            background-color: #ffffff;  /* Set background to white */
            color: #333333;  /* Set text color to dark for visibility */
            border-radius: 10px;
            border: 1px solid #ddd;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 80%;
            margin: 10px auto;
        }
    </style>
""", unsafe_allow_html=True)

# Display the title
st.markdown('<div class="title">Tempus MedGPT - Trial Version Dev1033-Shili34</div>', unsafe_allow_html=True)

# Test phase access banner
st.markdown('<div class="test-phase-banner">Test Phase Preview: Access for Tempus AI Employees Only</div>', unsafe_allow_html=True)

# Create an input text box and submit button
prompt = st.text_input("Enter your prompt", key="input", help="Ask any medical-related questions.")

# Generate response when the user clicks the 'Submit' button
if prompt:
    # Generate response from the fine-tuned BioGPT model
    generated_response = generate_response(prompt)

    # Display the response with a typing effect
    response_placeholder = st.empty()

    # Add some delay for simulating typing effect
    with st.spinner('Generating response...'):
        # Simulate typing effect
        for i in range(len(generated_response)):
            response_placeholder.markdown(f'<div class="response-box">{generated_response[:i+1]}</div>', unsafe_allow_html=True)
            time.sleep(0.05)  # Adjust typing speed

        # After finishing the typing effect, display the full response
        response_placeholder.markdown(f'<div class="response-box">{generated_response}</div>', unsafe_allow_html=True)

# Footer with a note or disclaimer
st.markdown("""
    <div style="text-align: center; color: grey; font-size: 12px; margin-top: 50px;">
        <p>© 2024 Tempus AI. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
