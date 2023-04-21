import base64
import os
import requests
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image
from io import BytesIO

engine_id = "stable-diffusion-xl-beta-v2-2-2"
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
api_key = "sk-8D7F65W7q1JCqIVQweTZBHJAtn2DKtSG71LbsPR25JuEWtny"

if api_key is None:
    raise Exception("Missing Stability API key.")

st.title("Image-to-Image Generation using Stability API")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    width, height = input_image.size

    uploaded_file.seek(0)
    multipart_data = MultipartEncoder(
        fields={
            "text_prompts[0][text]": "A lighthouse on a cliff",
            "init_image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": "0.35",
            "cfg_scale": "7",
            "clip_guidance_preset": "FAST_BLUE",                        
            "samples": "1",
            "steps": "30",
        }
    )

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/image-to-image",
        headers={
            "Content-Type": multipart_data.content_type,
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        data=multipart_data,
    )

    if response.status_code != 200:
        st.error("An error occurred: " + str(response.text))
    else:
        data = response.json()
        generated_image = base64.b64decode(data["artifacts"][0]["base64"])
        st.image(generated_image, caption="Generated Image", use_column_width=True)
        st.write(generated_image)
else:
    st.warning("Please upload an image to generate a new one.")
