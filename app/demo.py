import gradio as gr
from src.transcribe import transcribe

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(label="Transcription"),
    title="Huzhou Dialect Speech Recognition",
    description="Upload audio to transcribe Huzhou dialect speech.",
)

if __name__ == "__main__":
    demo.launch()