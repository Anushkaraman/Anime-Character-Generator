# 🛠️ Install Dependencies
!pip install torch torchvision diffusers gradio --quiet

# 🚀 Import Required Libraries
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# 📦 Load Model
model_id = "xyn-ai/anything-v4.0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)
pipe = pipe.to(device)

# 🎨 Generate Function
def generate_anime_image(prompt):
    if not prompt.strip():
        return "❗ Please enter a valid description.", None
    image = pipe(prompt).images[0]
    return f"✅ Image generated for: {prompt}", image

# 🌐 Gradio Web App
with gr.Blocks(css=".gr-button {background-color: #007BFF !important; color: white; border-radius: 8px;}") as demo:
    gr.Markdown("""
        <div style="text-align: center; background: linear-gradient(to right, #00c6ff, #0072ff); padding: 30px; border-radius: 10px; color: white;">
            <h1 style="font-size: 3em;">🎨 Anime Character Generator</h1>
            <p style="font-size: 1.2em;">Generate custom anime-style characters just by typing a prompt!</p>
        </div>
    """)
    
    gr.Markdown("<hr>")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                placeholder="Enter the prompt here (e.g. a ninja girl with pink hair)...",
                label="Prompt",
                lines=2,
                max_lines=4
            )
            submit_btn = gr.Button("✨ Submit Prompt")
            status = gr.Textbox(label="Status")
        
        with gr.Column(scale=2):
            output_image = gr.Image(label="Generated Anime Character")

    submit_btn.click(fn=generate_anime_image, inputs=prompt, outputs=[status, output_image])

demo.launch(share=True)
