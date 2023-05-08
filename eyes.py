
from repository.local_repo.csv_repo import CSVProcessor, get_app_settings
import os
import replicate

app_settings = get_app_settings()

image = "https://replicate.delivery/pbxt/Lg2F6Rzv7aIpI1Z7QVOUVqhRtFs0FNKKTI8MseiT9YX0AMcIA/out-0.png"

model = "rossjillian/controlnet_1-1"

version = "fe97435bfd17881fadfb8e290ebbf172f5835ac2ee015509d9d66b61a24bc5d3"

prompt = "close-up of an eye. perfect, normal eye, reflection on eye, aesthetically pleasuing,  black pupil, extremely detailed, very sharp. DSLR. highly detailed. hd. stunningly beautiful, hd"

structure = "canny"

scale = 10

steps = 50

os.environ["REPLICATE_API_TOKEN"] = app_settings["replicate_com_api_key"]

inputs = {
    "image": image,
    "prompt": prompt,
    "structure": structure,
    "scale": scale,
    "steps": steps
    
}

output = version.predict(**inputs)