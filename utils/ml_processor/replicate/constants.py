from dataclasses import dataclass


@dataclass
class ReplicateModel:
    name: str
    version: str

class REPLICATE_MODEL:
    andreas_sd_inpainting = ReplicateModel("andreasjansson/stable-diffusion-inpainting", "e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180")
    clones_lora_training = ReplicateModel("cloneofsimo/lora-training", "b2a308762e36ac48d16bfadc03a65493fe6e799f429f7941639a6acec5b276cc")
    clones_lora_training_2 = ReplicateModel("cloneofsimo/lora", "fce477182f407ffd66b94b08e761424cabd13b82b518754b83080bc75ad32466")
    google_frame_interpolation = ReplicateModel("google-research/frame-interpolation", "4f88a16a13673a8b589c18866e540556170a5bcb2ccdc12de556e800e9456d3d")
    pollination_modnet = ReplicateModel("pollinations/modnet", "da7d45f3b836795f945f221fc0b01a6d3ab7f5e163f13208948ad436001e2255")
    clip_interrogator = ReplicateModel("pharmapsychotic/clip-interrogator", "a4a8bafd6089e1716b06057c42b19378250d008b80fe87caa5cd36d40c1eda90")
    gfp_gan = ReplicateModel("xinntao/gfpgan", "6129309904ce4debfde78de5c209bce0022af40e197e132f08be8ccce3050393")
    ghost_face_swap = ReplicateModel("arielreplicate/ghost_face_swap", "106df0aaf9690354379d8cd291ad337f6b3ea02fe07d90feb1dafd64820066fa")
    stylegan_nada = ReplicateModel("rinongal/stylegan-nada", "6b2af4ac56fa2384f8f86fc7620943d5fc7689dcbb6183733743a215296d0e30")
    img2img_sd_2_1 = ReplicateModel("cjwbw/stable-diffusion-img2img-v2.1", "650c347f19a96c8a0379db998c4cd092e0734534591b16a60df9942d11dec15b")
    cjwbw_style_hair = ReplicateModel("cjwbw/style-your-hair", "c4c7e5a657e2e1abccd57625093522a9928edeccee77e3f55d57c664bcd96fa2")
    depth2img_sd = ReplicateModel("jagilley/stable-diffusion-depth2img", "68f699d395bc7c17008283a7cef6d92edc832d8dc59eb41a6cafec7fc70b85bc")
    salesforce_blip_2 = ReplicateModel("salesforce/blip-2", "4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
    phamquiluan_face_recognition = ReplicateModel("phamquiluan/facial-expression-recognition", "b16694d5bfed43612f1bfad7015cf2b7883b732651c383fe174d4b7783775ff5")
    arielreplicate = ReplicateModel("arielreplicate/instruct-pix2pix", "10e63b0e6361eb23a0374f4d9ee145824d9d09f7a31dcd70803193ebc7121430")
    cjwbw_midas = ReplicateModel("cjwbw/midas", "a6ba5798f04f80d3b314de0f0a62277f21ab3503c60c84d4817de83c5edfdae0")
    jagilley_controlnet_normal = ReplicateModel("jagilley/controlnet-normal", "cc8066f617b6c99fdb134bc1195c5291cf2610875da4985a39de50ee1f46d81c")
    jagilley_controlnet_canny = ReplicateModel("jagilley/controlnet-canny", "aff48af9c68d162388d230a2ab003f68d2638d88307bdaf1c2f1ac95079c9613")
    jagilley_controlnet_hed = ReplicateModel("jagilley/controlnet-hed", "cde353130c86f37d0af4060cd757ab3009cac68eb58df216768f907f0d0a0653")
    jagilley_controlnet_scribble = ReplicateModel("jagilley/controlnet-scribble", "435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117")
    jagilley_controlnet_seg = ReplicateModel("jagilley/controlnet-seg", "f967b165f4cd2e151d11e7450a8214e5d22ad2007f042f2f891ca3981dbfba0d")
    jagilley_controlnet_hough = ReplicateModel("jagilley/controlnet-hough", "854e8727697a057c525cdb45ab037f64ecca770a1769cc52287c2e56472a247b")
    jagilley_controlnet_depth2img = ReplicateModel("jagilley/controlnet-depth2img", "922c7bb67b87ec32cbc2fd11b1d5f94f0ba4f5519c4dbd02856376444127cc60")
    jagilley_controlnet_pose = ReplicateModel("jagilley/controlnet-pose", "0304f7f774ba7341ef754231f794b1ba3d129e3c46af3022241325ae0c50fb99")
    real_esrgan_upscale = ReplicateModel("cjwbw/real-esrgan", "d0ee3d708c9b911f122a4ad90046c5d26a0293b99476d697f6bb7f2e251ce2d4")


DEFAULT_LORA_MODEL_URL = "https://replicate.delivery/pbxt/nWm6eP9ojwVvBCaWoWZVawOKRfgxPJmkVk13ES7PX36Y66kQA/tmpxuz6k_k2datazip.safetensors"