from dataclasses import dataclass


@dataclass
class ReplicateModel:
    name: str
    version: str

class REPLICATE_MODEL:
    andreas_sd_inpainting = ReplicateModel("andreasjansson/stable-diffusion-inpainting", "e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180")
    clones_lora_training = ReplicateModel("cloneofsimo/lora-training", "b2a308762e36ac48d16bfadc03a65493fe6e799f429f7941639a6acec5b276cc")
    clones_lora_training_2 = ReplicateModel("cloneofsimo/lora", "fce477182f407ffd66b94b08e761424cabd13b82b518754b83080bc75ad32466")
    google_frame_interpolation = ReplicateModel("google-research/frame-interpolation", None)
    pollination_modnet = ReplicateModel("pollinations/modnet", None)
    clip_interrogator = ReplicateModel("pharmapsychotic/clip-interrogator", None)
    gfp_gan = ReplicateModel("xinntao/gfpgan", None)
    ghost_face_swap = ReplicateModel("arielreplicate/ghost_face_swap", "106df0aaf9690354379d8cd291ad337f6b3ea02fe07d90feb1dafd64820066fa")
    stylegan_nada = ReplicateModel("rinongal/stylegan-nada", None)
    img2img_sd_2_1 = ReplicateModel("cjwbw/stable-diffusion-img2img-v2.1", "650c347f19a96c8a0379db998c4cd092e0734534591b16a60df9942d11dec15b")
    cjwbw_style_hair = ReplicateModel("cjwbw/style-your-hair", "c4c7e5a657e2e1abccd57625093522a9928edeccee77e3f55d57c664bcd96fa2")
    depth2img_sd = ReplicateModel("jagilley/stable-diffusion-depth2img", "68f699d395bc7c17008283a7cef6d92edc832d8dc59eb41a6cafec7fc70b85bc")
    salesforce_blip_2 = ReplicateModel("salesforce/blip-2", "4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
    phamquiluan_face_recognition = ReplicateModel("phamquiluan/facial-expression-recognition", "b16694d5bfed43612f1bfad7015cf2b7883b732651c383fe174d4b7783775ff5")
    arielreplicate = ReplicateModel("arielreplicate/instruct-pix2pix", "10e63b0e6361eb23a0374f4d9ee145824d9d09f7a31dcd70803193ebc7121430")
    cjwbw_midas = ReplicateModel("cjwbw/midas", "a6ba5798f04f80d3b314de0f0a62277f21ab3503c60c84d4817de83c5edfdae0")
    jagilley_controlnet_normal = ReplicateModel("jagilley/controlnet-normal", None)
    jagilley_controlnet_canny = ReplicateModel("jagilley/controlnet-canny", None)
    jagilley_controlnet_hed = ReplicateModel("jagilley/controlnet-hed", None)
    jagilley_controlnet_scribble = ReplicateModel("jagilley/controlnet-scribble", None)
    jagilley_controlnet_seg = ReplicateModel("jagilley/controlnet-seg", None)
    jagilley_controlnet_hough = ReplicateModel("jagilley/controlnet-hough", None)
    jagilley_controlnet_depth2img = ReplicateModel("jagilley/controlnet-depth2img", None)
    jagilley_controlnet_pose = ReplicateModel("jagilley/controlnet-pose", None)


DEFAULT_LORA_MODEL_URL = "https://replicate.delivery/pbxt/nWm6eP9ojwVvBCaWoWZVawOKRfgxPJmkVk13ES7PX36Y66kQA/tmpxuz6k_k2datazip.safetensors"