import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.model_definitions import Generator, Uppscaling
import hashlib

# Carregamento dos modelos
MODEL_DIR = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelG = Generator().to(device)
modelG.load_state_dict(torch.load(f"{MODEL_DIR}/G_v1.pth", map_location=device))

modelU = Uppscaling().to(device)
modelU.load_state_dict(torch.load(f"{MODEL_DIR}/U_v1.pth", map_location=device))

def to_pil_image(tensor):
    tensor = tensor.squeeze(0)
    tensor = 0.5 * (tensor + 1.0)
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor)

norm = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def unnorm(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def text_to_seed(text):
    hash_object = hashlib.md5(text.encode())
    return int(hash_object.hexdigest(), 16)

# Interface Streamlit
st.title('Gerador e Upscaler de Imagens')

seed_text = st.text_input("Digite a seed para geração da imagem:", "texto_exemplo")
generate_button = st.button("Gerar Imagem")

if generate_button:
    # Geração da imagem baseada na seed
    num_seed = text_to_seed(seed_text) % (2**32)
    torch.manual_seed(num_seed)
    fixed_noise = torch.randn(1, 100, 1, 1, device=device)
    fake = modelG(fixed_noise).detach().cpu()
    img = unnorm(fake, *norm).squeeze(0)

    # Exibição da imagem original
    #st.image(Image.fromarray(np.transpose(img.numpy(), (1, 2, 0))), caption="Imagem Original")

    # Upscaling da imagem
    fake = fake.to(device)
    with torch.no_grad():
        output_image = modelU(fake)
    upscaled_image = to_pil_image(output_image.cpu())

    # Exibição da imagem upscalada
    st.image(upscaled_image, caption="Imagem")
