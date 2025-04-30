import gradio as gr
import requests

def ChatWithLLM(message):
    print(f"input from user: {message}")
    # Simulating a response for now
    return f"You said: {message}"

iface = gr.Interface(
    fn=ChatWithLLM,
    inputs="text",
    outputs="text",
    title="Testing the LLM using Ollama"
)

iface.launch(share=False)
