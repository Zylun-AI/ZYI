import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download, EntryNotFoundError

# ======== CONFIGURAÇÕES ========
REPO_ID = "caikybaldo999/ZYI"   # substitua pelo seu repo no Hugging Face
FILENAME = "ZYI.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======== FUNÇÃO PARA BAIXAR ========
def download_model():
    print(f"🔄 Baixando modelo '{FILENAME}' do repositório '{REPO_ID}'...")
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print("✅ Download concluído!")
        return model_path
    except EntryNotFoundError:
        raise RuntimeError(f"❌ Arquivo '{FILENAME}' não encontrado no Hugging Face ({REPO_ID}).")
    except Exception as e:
        raise RuntimeError(f"❌ Erro inesperado: {e}")

# ======== FUNÇÃO PARA CARREGAR ========
def load_model():
    model_path = download_model()
    print("🔧 Carregando modelo...")
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    pipe = pipe.to(DEVICE)
    print("✅ Modelo pronto para uso!")
    return pipe

# ======== INTERFACE GRADIO ========
def generate_image(prompt):
    try:
        image = pipe(prompt).images[0]
        return image
    except Exception as e:
        return f"Erro ao gerar imagem: {e}"

# Inicializa modelo
pipe = load_model()

# Cria interface
title = "ZYI Image Generator"
desc = "Gere imagens com o modelo ZYI"

interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Prompt", placeholder="Digite uma descrição..."),
    outputs=gr.Image(label="Imagem Gerada"),
    title=title,
    description=desc,
    examples=[
        ["a futuristic city made of bananas"],
        ["a realistic portrait of a cyberpunk warrior"],
        ["a fantasy forest with glowing trees at night"]
    ]
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
