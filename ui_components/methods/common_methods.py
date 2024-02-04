import io
from typing import List
import os
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from moviepy.editor import *
import cv2
import requests as r
import math
import json
import time
import uuid
from io import BytesIO
import numpy as np
import urllib3
from shared.constants import OFFLINE_MODE, SERVER, InferenceType, InternalFileTag, InternalFileType, ProjectMetaData, ServerType
from pydub import AudioSegment
from backend.models import InternalFileObject
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from ui_components.constants import SECOND_MASK_FILE, SECOND_MASK_FILE_PATH, WorkflowStageType
from ui_components.methods.file_methods import add_temp_file_to_project, convert_bytes_to_file, generate_pil_image, generate_temp_file, save_or_host_file, save_or_host_file_bytes
from ui_components.methods.video_methods import sync_audio_and_duration, update_speed_of_video_clip
from ui_components.models import InternalFrameTimingObject, InternalSettingObject
from utils.common_utils import acquire_lock, release_lock
from utils.data_repo.data_repo import DataRepo
from shared.constants import AnimationStyleType

from ui_components.models import InternalFileObject
from typing import Union

from utils.media_processor.video import VideoProcessor




def clone_styling_settings(source_frame_number, target_frame_uuid):
    data_repo = DataRepo()
    target_timing = data_repo.get_timing_from_uuid(target_frame_uuid)
    timing_list = data_repo.get_timing_list_from_shot(
        target_timing.shot.uuid)
    
    source_timing = timing_list[source_frame_number]
    params = source_timing.primary_image.inference_params

    if params:
        target_timing.prompt = params['prompt'] if 'prompt' in params else source_timing.prompt
        target_timing.negative_prompt = params['negative_prompt'] if 'negative_prompt' in params else source_timing.negative_prompt
        target_timing.guidance_scale = params['guidance_scale'] if 'guidance_scale' in params else source_timing.guidance_scale
        target_timing.seed = params['seed'] if 'seed' in params else source_timing.seed
        target_timing.num_inference_steps = params['num_inference_steps'] if 'num_inference_steps' in params else source_timing.num_inference_steps
        target_timing.strength = params['strength'] if 'strength' in params else source_timing.strength
        target_timing.adapter_type = params['adapter_type'] if 'adapter_type' in params else source_timing.adapter_type
        target_timing.low_threshold = params['low_threshold'] if 'low_threshold' in params else source_timing.low_threshold
        target_timing.high_threshold = params['high_threshold'] if 'high_threshold' in params else source_timing.high_threshold
    
        if 'model_uuid' in params and params['model_uuid']:
            model = data_repo.get_ai_model_from_uuid(params['model_uuid'])
            target_timing.model = model

# TODO: image format is assumed to be PNG, change this later
def save_new_image(img: Union[Image.Image, str, np.ndarray, io.BytesIO], project_uuid) -> InternalFileObject:
    data_repo = DataRepo()
    img = generate_pil_image(img)
    
    file_name = str(uuid.uuid4()) + ".png"
    file_path = os.path.join("videos/temp", file_name)

    hosted_url = save_or_host_file(img, file_path)

    file_data = {
        "name": str(uuid.uuid4()) + ".png",
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': file_path})
    
    new_image = data_repo.create_file(**file_data)
    return new_image

def save_and_promote_image(image, shot_uuid, timing_uuid, stage):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    try:
        saved_image = save_new_image(image, shot.project.uuid)
        # Update records based on stage
        if stage == WorkflowStageType.SOURCE.value:
            data_repo.update_specific_timing(timing_uuid, source_image_id=saved_image.uuid)
        elif stage == WorkflowStageType.STYLED.value:
            number_of_image_variants = add_image_variant(saved_image.uuid, timing_uuid)
            promote_image_variant(timing_uuid, number_of_image_variants - 1)

        return saved_image
    except Exception as e:
        print(f"Failed to save image file due to: {str(e)}")
        return None

def create_alpha_mask(size, edge_blur_radius):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)

    width, height = size
    draw.rectangle((edge_blur_radius, edge_blur_radius, width -
                   edge_blur_radius, height - edge_blur_radius), fill=255)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_blur_radius))
    return mask

# returns a PIL Image object
def zoom_image(image, zoom_factor, fill_with=None):
    blur_radius = 5
    edge_blur_radius = 15

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0")

    # Calculate new dimensions based on zoom factor
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    if zoom_factor < 1:
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        if fill_with == "Blur":
            blurred_image = image.filter(
                ImageFilter.GaussianBlur(radius=blur_radius))

            # Resize the blurred image to match the original dimensions
            blurred_background = blurred_image.resize(
                (width, height), Image.ANTIALIAS)

            # Create an alpha mask for blending
            alpha_mask = create_alpha_mask(
                resized_image.size, edge_blur_radius)

            # Blend the resized image with the blurred background using the alpha mask
            blended_image = Image.composite(resized_image, blurred_background.crop(
                (0, 0, new_width, new_height)), alpha_mask)

            # Calculate the position to paste the blended image at the center of the blurred background
            paste_left = (blurred_background.width - blended_image.width) // 2
            paste_top = (blurred_background.height - blended_image.height) // 2

            # Create a new blank image with the size of the blurred background
            final_image = Image.new('RGBA', blurred_background.size)

            # Paste the blurred background onto the final image
            final_image.paste(blurred_background, (0, 0))

            # Paste the blended image onto the final image using the alpha mask
            final_image.paste(blended_image, (paste_left,
                              paste_top), mask=alpha_mask)

            return final_image

        elif fill_with == "Inpainting":
            print("Coming soon")
            return resized_image

        elif fill_with is None:
            # Create an empty background with the original dimensions
            background = Image.new('RGBA', (width, height))

            # Calculate the position to paste the resized image at the center of the background
            paste_left = (background.width - resized_image.width) // 2
            paste_top = (background.height - resized_image.height) // 2

            # Paste the resized image onto the background
            background.paste(resized_image, (paste_left, paste_top))

            return background

        else:
            raise ValueError(
                "Invalid fill_with value. Accepted values are 'Blur', 'Inpainting', and None.")

    else:
        # If zooming in, proceed as before
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        left = (resized_image.width - width) / 2
        top = (resized_image.height - height) / 2
        right = (resized_image.width + width) / 2
        bottom = (resized_image.height + height) / 2

        cropped_image = resized_image.crop((left, top, right, bottom))
        return cropped_image

# image here is a PIL object
def apply_image_transformations(image: Image, zoom_level, rotation_angle, x_shift, y_shift, flip_vertically, flip_horizontally) -> Image:
    width, height = image.size

    # Calculate the diagonal for the rotation
    diagonal = math.ceil(math.sqrt(width**2 + height**2))

    # Create a new image with black background for rotation
    rotation_bg = Image.new("RGB", (diagonal, diagonal), "black")
    rotation_offset = ((diagonal - width) // 2, (diagonal - height) // 2)
    rotation_bg.paste(image, rotation_offset)

    # Rotation
    rotated_image = rotation_bg.rotate(rotation_angle)

    # Shift
    # Create a new image with black background
    shift_bg = Image.new("RGB", (diagonal, diagonal), "black")
    shift_bg.paste(rotated_image, (-x_shift, y_shift)) 

    # Zoom
    zoomed_width = int(diagonal * (zoom_level / 100))
    zoomed_height = int(diagonal * (zoom_level / 100))
    zoomed_image = shift_bg.resize((zoomed_width, zoomed_height))

    # Crop the zoomed image back to original size
    crop_x1 = (zoomed_width - width) // 2
    crop_y1 = (zoomed_height - height) // 2
    crop_x2 = crop_x1 + width
    crop_y2 = crop_y1 + height
    cropped_image = zoomed_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Flip vertically
    if flip_vertically:
        cropped_image = cropped_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Flip horizontally
    if flip_horizontally:
        cropped_image = cropped_image.transpose(Image.FLIP_LEFT_RIGHT)

    return cropped_image

def fetch_image_by_stage(shot_uuid, stage, frame_idx):
    data_repo = DataRepo()
    timing_list = data_repo.get_timing_list_from_shot(shot_uuid)

    if stage == WorkflowStageType.SOURCE.value:
        return timing_list[frame_idx].source_image
    elif stage == WorkflowStageType.STYLED.value:
        return timing_list[frame_idx].primary_image
    else:
        return None


# returns a PIL image object
def rotate_image(location, degree):
    if location.startswith('http') or location.startswith('https'):
        response = r.get(location)
        image = Image.open(BytesIO(response.content))
    else:
        if not os.path.exists(location):
            raise FileNotFoundError(f"File not found: {location}")
        image = Image.open(location)

    # Rotate the image by the specified degree
    rotated_image = image.rotate(-degree, resample=Image.BICUBIC, expand=False)

    return rotated_image


def save_uploaded_image(image: Union[Image.Image, str, np.ndarray, io.BytesIO, InternalFileObject], project_uuid, frame_uuid=None, stage_type=None):
    '''
    saves the image file (which can be a PIL, arr, InternalFileObject or url) into the project, without
    any tags or logs. then adds that file as the source_image/primary_image, depending
    on the stage selected
    '''
    data_repo = DataRepo()

    try:
        if isinstance(image, InternalFileObject):
            saved_image = image
        else:
            saved_image = save_new_image(image, project_uuid)
        
        # Update records based on stage_type
        if stage_type ==  WorkflowStageType.SOURCE.value:
            data_repo.update_specific_timing(frame_uuid, source_image_id=saved_image.uuid)
        elif stage_type ==  WorkflowStageType.STYLED.value:
            number_of_image_variants = add_image_variant(saved_image.uuid, frame_uuid)
            promote_image_variant(frame_uuid, number_of_image_variants - 1)

        return saved_image
    except Exception as e:
        print(f"Failed to save image file due to: {str(e)}")
        return None

# TODO: change variant_to_promote_frame_number to variant_uuid
def promote_image_variant(timing_uuid, variant_to_promote_frame_number: str):
    '''
    this methods promotes the variant to the primary image (also referred to as styled image)
    interpolated_clips/videos of the shot are not cleared
    '''
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    # promoting variant
    variant_to_promote = timing.alternative_images_list[variant_to_promote_frame_number]
    data_repo.update_specific_timing(timing_uuid, primary_image_id=variant_to_promote.uuid)
    _ = data_repo.get_timing_list_from_shot(timing.shot.uuid)


def promote_video_variant(shot_uuid, variant_uuid):
    '''
    this first changes the duration of the interpolated_clip to the frame clip_duration
    then adds the clip to the timed_clip (which is considered as the main variant)
    '''
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    variant_to_promote = None
    for variant in shot.interpolated_clip_list:
        if variant.uuid == variant_uuid:
            variant_to_promote = variant
            break
    
    if not variant_to_promote:
        return None

    if variant_to_promote.location.startswith(('http://', 'https://')):
        temp_video_path, _ = urllib3.request.urlretrieve(variant_to_promote.location)
        video = VideoFileClip(temp_video_path)
    else:
        video = VideoFileClip(variant_to_promote.location)

    if video.duration != shot.duration:
        video_bytes = VideoProcessor.update_video_speed(
            variant_to_promote.location,
            shot.duration
        )

        hosted_url = save_or_host_file_bytes(video_bytes, variant_to_promote.local_path)
        if hosted_url:
            data_repo.update_file(video.uuid, hosted_url=hosted_url)

    data_repo.update_shot(uuid=shot.uuid, main_clip_id=variant_to_promote.uuid)


def extract_canny_lines(image_path_or_url, project_uuid, low_threshold=50, high_threshold=150) -> InternalFileObject:
    data_repo = DataRepo()

    # Check if the input is a URL
    if image_path_or_url.startswith("http"):
        response = r.get(image_path_or_url)
        image_data = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    else:
        # Read the image from a local file
        image = cv2.imread(image_path_or_url, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply the Canny edge detection
    canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    # Reverse the colors (invert the image)
    inverted_canny_edges = 255 - canny_edges

    # Convert the inverted Canny edge result to a PIL Image
    new_canny_image = Image.fromarray(inverted_canny_edges)

    # Save the new image
    unique_file_name = str(uuid.uuid4()) + ".png"
    file_path = f"videos/{project_uuid}/assets/resources/masks/{unique_file_name}"
    hosted_url = save_or_host_file(new_canny_image, file_path)

    file_data = {
        "name": unique_file_name,
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': file_path})

    canny_image_file = data_repo.create_file(**file_data)
    return canny_image_file

# the input image is an image created by the PIL library
def create_or_update_mask(timing_uuid, image) -> InternalFileObject:
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    unique_file_name = str(uuid.uuid4()) + ".png"
    file_location = f"videos/{timing.shot.project.uuid}/assets/resources/masks/{unique_file_name}"

    hosted_url = save_or_host_file(image, file_location)
    # if mask is not present than creating a new one
    if not (timing.mask and timing.mask.location):
        file_data = {
            "name": unique_file_name,
            "type": InternalFileType.IMAGE.value
        }

        if hosted_url:
            file_data.update({'hosted_url': hosted_url})
        else:
            file_data.update({'local_path': file_location})

        mask_file: InternalFileObject = data_repo.create_file(**file_data)
        data_repo.update_specific_timing(timing_uuid, mask_id=mask_file.uuid)
    else:
        # if it is already present then just updating the file location
        if hosted_url:
            data_repo.update_file(timing.mask.uuid, hosted_url=hosted_url)
        else:
            data_repo.update_file(timing.mask.uuid, local_path=file_location)

    timing = data_repo.get_timing_from_uuid(timing_uuid)
    return timing.mask.location

def add_new_shot(project_uuid, name=""):
    data_repo = DataRepo()

    shot_data = {
        "project_uuid": project_uuid,
        "desc": "",
        "name": name,
        "duration": 10
    }

    shot = data_repo.create_shot(**shot_data)
    return shot

# adds the image file in variant (alternative images) list
def add_image_variant(image_file_uuid: str, timing_uuid: str):
    data_repo = DataRepo()
    image_file: InternalFileObject = data_repo.get_file_from_uuid(
        image_file_uuid)
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)

    alternative_image_list = timing.alternative_images_list + [image_file]
    alternative_image_uuid_list = [img.uuid for img in alternative_image_list]
    primary_image_uuid = alternative_image_uuid_list[0]
    alternative_image_uuid_list = json.dumps(alternative_image_uuid_list)

    data_repo.update_specific_timing(
        timing_uuid, alternative_images=alternative_image_uuid_list)

    if not timing.primary_image:
        data_repo.update_specific_timing(
            timing_uuid, primary_image_id=primary_image_uuid)

    return len(alternative_image_list)

# image_list is a list of uploaded_obj
def convert_image_list_to_file_list(image_list):
    data_repo = DataRepo()
    file_list = []
    for image in image_list:
        img = Image.open(image)
        filename = str(uuid.uuid4()) + ".png"
        file_path = "videos/training_data/" + filename
        hosted_url = save_or_host_file(img, file_path)
        data = {
            "name": str(uuid.uuid4()),
            "type": InternalFileType.IMAGE.value,
        }

        if hosted_url:
            data['hosted_url'] = hosted_url
        else:
            data['local_path'] = file_path

        image_file = data_repo.create_file(**data)
        file_list.append(image_file)
    return file_list

def replace_background(project_uuid, bg_img_loc) -> InternalFileObject:
    data_repo = DataRepo()
    project = data_repo.get_project_from_uuid(project_uuid)
    background_image = generate_pil_image(bg_img_loc)
    
    path = project.get_temp_mask_file(SECOND_MASK_FILE).location
    foreground_image = generate_pil_image(path)

    background_image.paste(foreground_image, (0, 0), foreground_image)
    filename = str(uuid.uuid4()) + ".png"
    background_img_path = f"videos/{project_uuid}/replaced_bg.png"
    hosted_url = save_or_host_file(background_image, background_img_path)
    file_data = {
        "name": filename,
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': background_img_path})
    
    image_file = data_repo.create_file(**file_data)

    return image_file

# TODO: don't save or upload image where just passing the PIL object can work
def resize_image(video_name, new_width, new_height, image_file: InternalFileObject) -> InternalFileObject:
    if 'http' in image_file.location:
        response = r.get(image_file.location)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file.location)
    resized_image = image.resize((new_width, new_height))

    time.sleep(0.1)

    unique_id = str(uuid.uuid4())
    filepath = "videos/" + str(video_name) + \
        "/temp_image-" + unique_id + ".png"
    
    hosted_url = save_or_host_file(resized_image, filepath)
    file_data = {
        "name": str(uuid.uuid4()) + ".png",
        "type": InternalFileType.IMAGE.value
    }

    if hosted_url:
        file_data.update({'hosted_url': hosted_url})
    else:
        file_data.update({'local_path': filepath})

    data_repo = DataRepo()
    image_file = data_repo.create_file(**file_data)

    return image_file

def get_audio_bytes_for_slice(timing_uuid):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(
        timing.shot.project.uuid)

    # TODO: add null check for the audio
    audio = AudioSegment.from_file(project_settings.audio.local_path)

    # DOUBT: is it checked if it is the last frame or not?
    audio = audio[timing.frame_time *
                  1000: data_repo.get_next_timing(timing_uuid)['frame_time'] * 1000]
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format='wav')
    audio_bytes.seek(0)
    return audio_bytes


def create_frame_inside_shot(shot_uuid, aux_frame_index):
    data_repo = DataRepo()
    
    timing_data = {
        "shot_id": shot_uuid,
        "animation_style": AnimationStyleType.CREATIVE_INTERPOLATION.value,
        "aux_frame_index": aux_frame_index
    }
    timing: InternalFrameTimingObject = data_repo.create_timing(**timing_data)

    return timing


def save_audio_file(uploaded_file, project_uuid):
    data_repo = DataRepo()

    local_file_location = os.path.join(
        f"videos/{project_uuid}/assets/resources/audio", uploaded_file.name)

    audio_bytes = uploaded_file.read()
    hosted_url = save_or_host_file_bytes(audio_bytes, local_file_location, ".mp3")

    file_data = {
        "name": str(uuid.uuid4()) + ".mp3",
        "type": InternalFileType.AUDIO.value,
        "project_id": project_uuid

    }

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": local_file_location})

    audio_file: InternalFileObject = data_repo.create_file(
        **file_data)
    data_repo.update_project_setting(
        project_uuid, audio_id=audio_file.uuid)
    
    return audio_file

def execute_image_edit(type_of_mask_selection, type_of_mask_replacement,
                       background_image, editing_image, prompt, negative_prompt,
                       width, height, layer, timing_uuid):
    from ui_components.methods.ml_methods import inpainting, remove_background, create_depth_mask_image
    data_repo = DataRepo()
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(
        timing_uuid)
    project = timing.shot.project
    inference_log = None

    if type_of_mask_selection == "Manual Background Selection":
        # NOTE: code not is use
        # if type_of_mask_replacement == "Replace With Image":
        #     bg_img = generate_pil_image(editing_image)
        #     mask_img = generate_pil_image(timing.mask.location)

        #     result_img = Image.new("RGBA", bg_img.size, (255, 255, 255, 0))
        #     for x in range(bg_img.size[0]):
        #         for y in range(bg_img.size[1]):
        #             if x < mask_img.size[0] and y < mask_img.size[1]:
        #                 if mask_img.getpixel((x, y)) == (255, 255, 255):
        #                     result_img.putpixel((x, y), (255, 255, 255, 0))
        #                 else:
        #                     result_img.putpixel((x, y), bg_img.getpixel((x, y)))
            
        #     hosted_manual_bg_url = save_or_host_file(result_img, SECOND_MASK_FILE_PATH)
        #     add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_manual_bg_url or SECOND_MASK_FILE_PATH)
        #     edited_image = replace_background(project.uuid, background_image)

        if type_of_mask_replacement == "Inpainting":
            edited_image, log = inpainting(editing_image, prompt, negative_prompt, timing_uuid, False)
            inference_log = log
    
    # NOTE: code not is use -------------------------------------
    # elif type_of_mask_selection == "Automated Background Selection":
    #     removed_background = remove_background(editing_image)
    #     response = r.get(removed_background)
    #     img = Image.open(BytesIO(response.content))
    #     hosted_url = save_or_host_file(img, SECOND_MASK_FILE_PATH)
    #     add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_url or SECOND_MASK_FILE_PATH)

    #     if type_of_mask_replacement == "Replace With Image":
    #         edited_image = replace_background(project.uuid, background_image)

    #     elif type_of_mask_replacement == "Inpainting":
    #         path = project.get_temp_mask_file(SECOND_MASK_FILE).location
    #         if path.startswith("http"):
    #             response = r.get(path)
    #             image = Image.open(BytesIO(response.content))
    #         else:
    #             image = Image.open(path)

    #         converted_image = Image.new("RGB", image.size, (255, 255, 255))
    #         for x in range(image.width):
    #             for y in range(image.height):
    #                 pixel = image.getpixel((x, y))
    #                 if pixel[3] == 0:
    #                     converted_image.putpixel((x, y), (0, 0, 0))
    #                 else:
    #                     converted_image.putpixel((x, y), (255, 255, 255))
    #         create_or_update_mask(timing_uuid, converted_image)
    #         edited_image = inpainting(
    #             editing_image, prompt, negative_prompt, timing.uuid, True)
            
    # elif type_of_mask_selection == "Automated Layer Selection":
    #     mask_location = create_depth_mask_image(
    #         editing_image, layer, timing.uuid)
    #     if type_of_mask_replacement == "Replace With Image":
    #         if mask_location.startswith("http"):
    #             mask = Image.open(
    #                 BytesIO(r.get(mask_location).content)).convert('1')
    #         else:
    #             mask = Image.open(mask_location).convert('1')
    #         if editing_image.startswith("http"):
    #             response = r.get(editing_image)
    #             bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
    #         else:
    #             bg_img = Image.open(editing_image).convert('RGBA')

    #         hosted_automated_bg_url = save_or_host_file(result_img, SECOND_MASK_FILE_PATH)
    #         add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_automated_bg_url or SECOND_MASK_FILE_PATH)
    #         edited_image = replace_background(project.uuid, SECOND_MASK_FILE_PATH, background_image)

    #     elif type_of_mask_replacement == "Inpainting":
    #         edited_image = inpainting(
    #             editing_image, prompt, negative_prompt, timing_uuid, True)

    # elif type_of_mask_selection == "Re-Use Previous Mask":
    #     mask_location = timing.mask.location
    #     if type_of_mask_replacement == "Replace With Image":
    #         if mask_location.startswith("http"):
    #             response = r.get(mask_location)
    #             mask = Image.open(BytesIO(response.content)).convert('1')
    #         else:
    #             mask = Image.open(mask_location).convert('1')
    #         if editing_image.startswith("http"):
    #             response = r.get(editing_image)
    #             bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
    #         else:
    #             bg_img = Image.open(editing_image).convert('RGBA')
            
    #         hosted_image_replace_url = save_or_host_file(result_img, SECOND_MASK_FILE_PATH)
    #         add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_image_replace_url or SECOND_MASK_FILE_PATH)
    #         edited_image = replace_background(project.uuid, SECOND_MASK_FILE_PATH, background_image)

    #     elif type_of_mask_replacement == "Inpainting":
    #         edited_image = inpainting(
    #             editing_image, prompt, negative_prompt, timing_uuid, True)

    # elif type_of_mask_selection == "Invert Previous Mask":
    #     if type_of_mask_replacement == "Replace With Image":
    #         mask_location = timing.mask.location
    #         if mask_location.startswith("http"):
    #             response = r.get(mask_location)
    #             mask = Image.open(BytesIO(response.content)).convert('1')
    #         else:
    #             mask = Image.open(mask_location).convert('1')
    #         inverted_mask = ImageOps.invert(mask)
    #         if editing_image.startswith("http"):
    #             response = r.get(editing_image)
    #             bg_img = Image.open(BytesIO(response.content)).convert('RGBA')
    #         else:
    #             bg_img = Image.open(editing_image).convert('RGBA')
    #         masked_img = Image.composite(bg_img, Image.new(
    #             'RGBA', bg_img.size, (0, 0, 0, 0)), inverted_mask)
    #         # TODO: standardise temproray fixes
    #         hosted_prvious_invert_url = save_or_host_file(result_img, SECOND_MASK_FILE_PATH)
    #         add_temp_file_to_project(project.uuid, SECOND_MASK_FILE, hosted_prvious_invert_url or SECOND_MASK_FILE_PATH)
    #         edited_image = replace_background(project.uuid, SECOND_MASK_FILE_PATH, background_image)

    #     elif type_of_mask_replacement == "Inpainting":
    #         edited_image = inpainting(
    #             editing_image, prompt, negative_prompt, timing_uuid, False)
    # ---------------------------------------------------------------------

    return edited_image, inference_log


# if the output is present it adds it to the respective place or else it updates the inference log
# NOTE: every function used in this should not change/modify session state in anyway
def process_inference_output(**kwargs):
    data_repo = DataRepo()

    inference_time = 0.0
    inference_type = kwargs.get('inference_type')
    log_uuid = None
    # ------------------- FRAME TIMING IMAGE INFERENCE -------------------
    if inference_type == InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value:
        output = kwargs.get('output')
        if output:
            timing_uuid = kwargs.get('timing_uuid')
            promote_new_generation = kwargs.get('promote_new_generation')

            timing = data_repo.get_timing_from_uuid(timing_uuid)
            if not timing:
                return False
            
            filename = str(uuid.uuid4()) + ".png"
            log_uuid = kwargs.get('log_uuid')
            log = data_repo.get_inference_log_from_uuid(log_uuid)
            if log and log.total_inference_time:
                inference_time = log.total_inference_time

            output_file = data_repo.create_file(
                name=filename, 
                type=InternalFileType.IMAGE.value,
                hosted_url=output[0] if isinstance(output, list) else output, 
                inference_log_id=log.uuid,
                project_id=timing.shot.project.uuid,
            )
            
            add_image_variant(output_file.uuid, timing_uuid)
            if promote_new_generation == True:
                timing = data_repo.get_timing_from_uuid(timing_uuid)
                variants = timing.alternative_images_list
                number_of_variants = len(variants)
                if number_of_variants == 1:
                    print("No new generation to promote")
                else:
                    promote_image_variant(timing_uuid, number_of_variants - 1)
            else:
                print("No new generation to promote")
        else:
            log_uuid = kwargs.get('log_uuid')
            del kwargs['log_uuid']
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)
    
    # --------------------- MULTI VIDEO INFERENCE (INTERPOLATION + MORPHING) -------------------
    elif inference_type == InferenceType.FRAME_INTERPOLATION.value:
        output = kwargs.get('output')
        log_uuid = kwargs.get('log_uuid')

        if output:
            settings = kwargs.get('settings')
            shot_uuid = kwargs.get('shot_uuid')
            shot = data_repo.get_shot_from_uuid(shot_uuid)
            if not shot:
                return False
            
            # output can also be an url
            if isinstance(output, str) and output.startswith("http"):
                temp_output_file = generate_temp_file(output, '.mp4')
                output = None
                with open(temp_output_file.name, 'rb') as f:
                    output = f.read()

                os.remove(temp_output_file.name)

            if 'normalise_speed' in settings and settings['normalise_speed']:
                output = VideoProcessor.update_video_bytes_speed(output, shot.duration)

            video_location = "videos/" + str(shot.project.uuid) + "/assets/videos/0_raw/" + str(uuid.uuid4()) + ".mp4"
            video = convert_bytes_to_file(
                file_location_to_save=video_location,
                mime_type="video/mp4",
                file_bytes=output,
                project_uuid=shot.project.uuid,
                inference_log_id=log_uuid
            )

            if not shot.main_clip:
                output_video = sync_audio_and_duration(video, shot_uuid)
                data_repo.update_shot(uuid=shot_uuid, main_clip_id=output_video.uuid)
                data_repo.add_interpolated_clip(shot_uuid, interpolated_clip_id=output_video.uuid)
            else:
                data_repo.add_interpolated_clip(shot_uuid, interpolated_clip_id=video.uuid)

            log = data_repo.get_inference_log_from_uuid(log_uuid)
            if log and log.total_inference_time:
                inference_time = log.total_inference_time
        else:
            del kwargs['log_uuid']
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    # --------------------- GALLERY IMAGE GENERATION ------------------------
    elif inference_type == InferenceType.GALLERY_IMAGE_GENERATION.value:
        output = kwargs.get('output')

        if output:
            log_uuid = kwargs.get('log_uuid')
            project_uuid = kwargs.get('project_uuid')
            log = data_repo.get_inference_log_from_uuid(log_uuid)
            if log and log.total_inference_time:
                inference_time = log.total_inference_time

            filename = str(uuid.uuid4()) + ".png"
            output_file = data_repo.create_file(
                name=filename, 
                type=InternalFileType.IMAGE.value,
                hosted_url=output[0] if isinstance(output, list) else output, 
                inference_log_id=log.uuid,
                project_id=project_uuid,
                tag=InternalFileTag.TEMP_GALLERY_IMAGE.value        # will be updated to GALLERY_IMAGE once the user clicks 'check for new images'
            )
        else:
            log_uuid = kwargs.get('log_uuid')
            del kwargs['log_uuid']
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    # --------------------- FRAME INPAINTING ------------------------
    elif inference_type == InferenceType.FRAME_INPAINTING.value:
        output = kwargs.get('output')
        log_uuid = kwargs.get('log_uuid')

        if output:
            stage = kwargs.get('stage', WorkflowStageType.STYLED.value)
            promote = kwargs.get('promote_generation', False)
            current_frame_uuid = kwargs.get('timing_uuid')
            timing = data_repo.get_timing_from_uuid(current_frame_uuid)

            file_name = str(uuid.uuid4()) + ".png"
            output_file = data_repo.create_file(
                name=file_name, 
                type=InternalFileType.IMAGE.value, 
                hosted_url=output[0] if isinstance(output, list) else output, 
                inference_log_id=str(log_uuid),
                project_id=timing.shot.project.uuid
            )

            if stage == WorkflowStageType.SOURCE.value:
                data_repo.update_specific_timing(current_frame_uuid, source_image_id=output_file.uuid)
            elif stage == WorkflowStageType.STYLED.value:
                number_of_image_variants = add_image_variant(output_file.uuid, current_frame_uuid)
                if promote:
                    promote_image_variant(current_frame_uuid, number_of_image_variants - 1)
            
            log = data_repo.get_inference_log_from_uuid(log_uuid)
            if log and log.total_inference_time:
                inference_time = log.total_inference_time
        else:
            del kwargs['log_uuid']
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    if inference_time:
        credits_used = round(inference_time * 0.004, 3)     # make this more granular for different models
        data_repo.update_usage_credits(-credits_used, log_uuid)

    return True


def check_project_meta_data(project_uuid):
    '''
    checking for project metadata (like cache updates - we update specific entities using this flag)
    project_update_data is of the format {"data_update": [timing_uuid], "gallery_update": True/False, "background_img_list": []}
    '''
    data_repo = DataRepo()
    
    key = project_uuid
    if acquire_lock(key):
        project = data_repo.get_project_from_uuid(project_uuid)
        timing_update_data = json.loads(project.meta_data).\
            get(ProjectMetaData.DATA_UPDATE.value, None) if project.meta_data else None
        if timing_update_data and len(timing_update_data):
            for timing_uuid in timing_update_data:
                _ = data_repo.get_timing_from_uuid(timing_uuid, invalidate_cache=True)

        gallery_update_data = json.loads(project.meta_data).\
            get(ProjectMetaData.GALLERY_UPDATE.value, False) if project.meta_data else False
        if gallery_update_data:
            pass

        shot_update_data = json.loads(project.meta_data).\
            get(ProjectMetaData.SHOT_VIDEO_UPDATE.value, []) if project.meta_data else []
        if shot_update_data and len(shot_update_data):
            for shot_uuid in shot_update_data:
                _ = data_repo.get_shot_list(shot_uuid, invalidate_cache=True)
        
        # clearing update data from cache
        meta_data = {
            ProjectMetaData.DATA_UPDATE.value: [],
            ProjectMetaData.GALLERY_UPDATE.value: False,
            ProjectMetaData.SHOT_VIDEO_UPDATE.value: []
        }
        data_repo.update_project(uuid=project.uuid, meta_data=json.dumps(meta_data))
        
        release_lock(key)


def update_app_setting_keys():
    data_repo = DataRepo()
    app_logger = AppLogger()

    if OFFLINE_MODE:
        key = os.getenv('REPLICATE_KEY', None)
    else:
        import boto3
        ssm = boto3.client("ssm", region_name="ap-south-1")
        key = ssm.get_parameter(Name='/backend/banodoco/replicate/key')['Parameter']['Value']

    app_setting = data_repo.get_app_secrets_from_user_uuid()
    if app_setting and app_setting['replicate_key'] == key:
        return

    app_logger.log(LoggingType.DEBUG, 'setting keys', None)
    data_repo.update_app_setting(replicate_username='bn')
    data_repo.update_app_setting(replicate_key=key)    