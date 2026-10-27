"""
Hugging Face Spaces Entry Point
================================
Runs the unified Gradio inspection dashboard in cloud deployment.
"""
from app_gradio import demo

if __name__ == "__main__":
    demo.launch()
