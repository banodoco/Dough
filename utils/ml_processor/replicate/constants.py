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
    jagilley_controlnet_normal = ReplicateModel("jagilley/controlnet-normal", None)
    jagilley_controlnet_canny = ReplicateModel("jagilley/controlnet-canny", None)
    jagilley_controlnet_hed = ReplicateModel("jagilley/controlnet-hed", None)
    jagilley_controlnet_scribble = ReplicateModel("jagilley/controlnet-scribble", None)
    jagilley_controlnet_seg = ReplicateModel("jagilley/controlnet-seg", None)
    jagilley_controlnet_hough = ReplicateModel("jagilley/controlnet-hough", None)
    jagilley_controlnet_depth2img = ReplicateModel("jagilley/controlnet-depth2img", None)
    jagilley_controlnet_pose = ReplicateModel("jagilley/controlnet-pose", None)
    real_esrgan_upscale = ReplicateModel("cjwbw/real-esrgan", "d0ee3d708c9b911f122a4ad90046c5d26a0293b99476d697f6bb7f2e251ce2d4")
    controlnet_1_1_x_realistic_vision_v2_0 = ReplicateModel("usamaehsan/controlnet-1.1-x-realistic-vision-v2.0", "7fbf4c86671738f97896c9cb4922705adfcdcf54a6edab193bb8c176c6b34a69")
    urpm = ReplicateModel("mcai/urpm-v1.3-img2img", "4df956e8dbfebf1afaf0c3ee98ad426ec58c4262d24360d054582e5eab2cb5f6")
    sdxl = ReplicateModel("stability-ai/sdxl", "af1a68a271597604546c09c64aabcd7782c114a63539a4a8d14d1eeda5630c33")

    # addition 30/9/2023
    realistic_vision_v5 = ReplicateModel("heedster/realistic-vision-v5", "c0259010b93e7a4102a4ba946d70e06d7d0c7dc007201af443cfc8f943ab1d3c")
    deliberate_v3 = ReplicateModel("pagebrain/deliberate-v3", "1851b62340ae657f05f8b8c8a020e3f9a46efde9fe80f273eef026c0003252ac")
    dreamshaper_v7 = ReplicateModel("pagebrain/dreamshaper-v7", "0deba88df4e49b302585e1a7b6bd155e18962c1048966a40fe60ba05805743ff")
    epicrealism_v5 = ReplicateModel("pagebrain/epicrealism-v5", "222465e57e4d9812207f14133c9499d47d706ecc41a8bf400120285b2f030b42")
    sdxl_controlnet = ReplicateModel("lucataco/sdxl-controlnet", "db2ffdbdc7f6cb4d6dab512434679ee3366ae7ab84f89750f8947d5594b79a47")
    realistic_vision_v5_img2img = ReplicateModel("lucataco/realistic-vision-v5-img2img", "82bbb4595458d6be142450fc6d8c4d79c936b92bd184dd2d6dd71d0796159819")

    @staticmethod
    def get_model_by_db_obj(model_db_obj):
        for model in REPLICATE_MODEL.__dict__.values():
            if isinstance(model, ReplicateModel) and model.name == model_db_obj.replicate_url and model.version == model_db_obj.version:
                return model
        return None

DEFAULT_LORA_MODEL_URL = "https://replicate.delivery/pbxt/nWm6eP9ojwVvBCaWoWZVawOKRfgxPJmkVk13ES7PX36Y66kQA/tmpxuz6k_k2datazip.safetensors"