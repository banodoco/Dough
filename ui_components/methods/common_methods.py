import io
import multiprocessing
import random
import sys
from typing import List
import os
from PIL import Image, ImageDraw, ImageFilter
from utils.common_utils import sqlite_atomic_transaction
from utils.local_storage.local_storage import write_to_motion_lora_local_db
from moviepy.editor import *
import cv2
import requests as r
import math
import json
import time
import uuid
from io import BytesIO
import numpy as np
from django.db import connection
from shared.constants import (
    COMFY_BASE_PATH,
    OFFLINE_MODE,
    SERVER,
    InferenceStatus,
    InferenceType,
    InternalFileTag,
    InternalFileType,
    ProjectMetaData,
)
from pydub import AudioSegment
from backend.models import InternalFileObject
from shared.logging.constants import LoggingType
from shared.logging.logging import AppLogger
from ui_components.constants import SECOND_MASK_FILE, WorkflowStageType
from ui_components.methods.file_methods import (
    convert_bytes_to_file,
    generate_pil_image,
    generate_temp_file,
    save_or_host_file,
    save_or_host_file_bytes,
)
from ui_components.methods.video_methods import sync_audio_and_duration
from ui_components.models import (
    InferenceLogObject,
    InternalFrameTimingObject,
    InternalProjectObject,
    InternalSettingObject,
)
from utils.data_repo.data_repo import DataRepo
from shared.constants import AnimationStyleType

from ui_components.models import InternalFileObject
from typing import Union

from utils.ml_processor.gpu.utils import COMFY_RUNNER_PATH


# TODO: image format is assumed to be PNG, change this later
def save_new_image(img: Union[Image.Image, str, np.ndarray, io.BytesIO], project_uuid) -> InternalFileObject:
    """
    Saves an image into the project. The image is not added into any shot and is without tags.
    """
    data_repo = DataRepo()
    img = generate_pil_image(img)

    file_name = str(uuid.uuid4()) + ".png"
    file_path = os.path.join("videos/temp", file_name)

    hosted_url = save_or_host_file(img, file_path)

    file_data = {
        "name": str(uuid.uuid4()) + ".png",
        "type": InternalFileType.IMAGE.value,
        "project_id": project_uuid,
    }

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": file_path})

    new_image = data_repo.create_file(**file_data)
    return new_image


def save_and_promote_image(image, shot_uuid, timing_uuid, stage):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    try:
        saved_image = save_new_image(image, shot.project.uuid)
        # Update records based on stage
        if stage == WorkflowStageType.SOURCE.value:
            data_repo.update_specific_timing(
                timing_uuid, source_image_id=saved_image.uuid, update_in_place=True
            )
        elif stage == WorkflowStageType.STYLED.value:
            number_of_image_variants = add_image_variant(saved_image.uuid, timing_uuid)
            promote_image_variant(timing_uuid, number_of_image_variants - 1)

        return saved_image
    except Exception as e:
        print(f"Failed to save image file due to: {str(e)}")
        return None


def create_alpha_mask(size, edge_blur_radius):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    width, height = size
    draw.rectangle(
        (edge_blur_radius, edge_blur_radius, width - edge_blur_radius, height - edge_blur_radius), fill=255
    )

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
            blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Resize the blurred image to match the original dimensions
            blurred_background = blurred_image.resize((width, height), Image.ANTIALIAS)

            # Create an alpha mask for blending
            alpha_mask = create_alpha_mask(resized_image.size, edge_blur_radius)

            # Blend the resized image with the blurred background using the alpha mask
            blended_image = Image.composite(
                resized_image, blurred_background.crop((0, 0, new_width, new_height)), alpha_mask
            )

            # Calculate the position to paste the blended image at the center of the blurred background
            paste_left = (blurred_background.width - blended_image.width) // 2
            paste_top = (blurred_background.height - blended_image.height) // 2

            # Create a new blank image with the size of the blurred background
            final_image = Image.new("RGBA", blurred_background.size)

            # Paste the blurred background onto the final image
            final_image.paste(blurred_background, (0, 0))

            # Paste the blended image onto the final image using the alpha mask
            final_image.paste(blended_image, (paste_left, paste_top), mask=alpha_mask)

            return final_image

        elif fill_with == "Inpainting":
            print("Coming soon")
            return resized_image

        elif fill_with is None:
            # Create an empty background with the original dimensions
            background = Image.new("RGBA", (width, height))

            # Calculate the position to paste the resized image at the center of the background
            paste_left = (background.width - resized_image.width) // 2
            paste_top = (background.height - resized_image.height) // 2

            # Paste the resized image onto the background
            background.paste(resized_image, (paste_left, paste_top))

            return background

        else:
            raise ValueError("Invalid fill_with value. Accepted values are 'Blur', 'Inpainting', and None.")

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
def apply_image_transformations(
    image: Image, zoom_level, rotation_angle, x_shift, y_shift, flip_vertically, flip_horizontally
) -> Image:
    width, height = image.size

    # Calculate the diagonal for the rotation
    diagonal = math.ceil(math.sqrt(width**2 + height**2))

    # Create a new image with white background for rotation
    rotation_bg = Image.new("RGB", (diagonal, diagonal), "white")
    rotation_offset = ((diagonal - width) // 2, (diagonal - height) // 2)
    rotation_bg.paste(image, rotation_offset)

    # Rotation - Rotate in the opposite direction
    rotated_image = rotation_bg.rotate(-rotation_angle)

    # Shift - Invert the direction of the shift
    shift_bg = Image.new("RGB", (diagonal, diagonal), "white")
    shift_bg.paste(rotated_image, (x_shift, -y_shift))

    # Zoom - Adjust zoom level
    zoomed_width = int(diagonal * (zoom_level / 100))
    zoomed_height = int(diagonal * (zoom_level / 100))
    zoomed_image = shift_bg.resize((zoomed_width, zoomed_height), Image.ANTIALIAS)

    # Create a new image with white background to accommodate the zoomed image
    final_image = Image.new("RGB", (width, height), "white")

    # Calculate the position to paste the zoomed image at the center of the final image
    paste_x = (width - zoomed_width) // 2
    paste_y = (height - zoomed_height) // 2

    # Paste the zoomed image onto the final image
    final_image.paste(zoomed_image, (paste_x, paste_y))

    # Flip vertically - No change
    if flip_vertically:
        final_image = final_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Flip horizontally - No change
    if flip_horizontally:
        final_image = final_image.transpose(Image.FLIP_LEFT_RIGHT)

    return final_image


def apply_coord_transformations(
    initial_coords, zoom_level, rotation_angle, x_shift, y_shift, flip_vertically, flip_horizontally
):
    x1, y1 = initial_coords[0]
    x2, y2 = initial_coords[1]
    x3, y3 = initial_coords[2]
    x4, y4 = initial_coords[3]

    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4

    # zoom
    x1_zoomed = center_x + zoom_level * (x1 - center_x) / 100
    y1_zoomed = center_y + zoom_level * (y1 - center_y) / 100
    x2_zoomed = center_x + zoom_level * (x2 - center_x) / 100
    y2_zoomed = center_y + zoom_level * (y2 - center_y) / 100
    x3_zoomed = center_x + zoom_level * (x3 - center_x) / 100
    y3_zoomed = center_y + zoom_level * (y3 - center_y) / 100
    x4_zoomed = center_x + zoom_level * (x4 - center_x) / 100
    y4_zoomed = center_y + zoom_level * (y4 - center_y) / 100

    # rotate
    rotation_angle_rad = math.radians(rotation_angle)
    x1_rotated = (
        center_x
        + math.cos(rotation_angle_rad) * (x1_zoomed - center_x)
        - math.sin(rotation_angle_rad) * (y1_zoomed - center_y)
    )
    y1_rotated = (
        center_y
        + math.sin(rotation_angle_rad) * (x1_zoomed - center_x)
        + math.cos(rotation_angle_rad) * (y1_zoomed - center_y)
    )
    x2_rotated = (
        center_x
        + math.cos(rotation_angle_rad) * (x2_zoomed - center_x)
        - math.sin(rotation_angle_rad) * (y2_zoomed - center_y)
    )
    y2_rotated = (
        center_y
        + math.sin(rotation_angle_rad) * (x2_zoomed - center_x)
        + math.cos(rotation_angle_rad) * (y2_zoomed - center_y)
    )
    x3_rotated = (
        center_x
        + math.cos(rotation_angle_rad) * (x3_zoomed - center_x)
        - math.sin(rotation_angle_rad) * (y3_zoomed - center_y)
    )
    y3_rotated = (
        center_y
        + math.sin(rotation_angle_rad) * (x3_zoomed - center_x)
        + math.cos(rotation_angle_rad) * (y3_zoomed - center_y)
    )
    x4_rotated = (
        center_x
        + math.cos(rotation_angle_rad) * (x4_zoomed - center_x)
        - math.sin(rotation_angle_rad) * (y4_zoomed - center_y)
    )
    y4_rotated = (
        center_y
        + math.sin(rotation_angle_rad) * (x4_zoomed - center_x)
        + math.cos(rotation_angle_rad) * (y4_zoomed - center_y)
    )

    # shift
    x1_shifted = x1_rotated + x_shift
    y1_shifted = y1_rotated + y_shift
    x2_shifted = x2_rotated + x_shift
    y2_shifted = y2_rotated + y_shift
    x3_shifted = x3_rotated + x_shift
    y3_shifted = y3_rotated + y_shift
    x4_shifted = x4_rotated + x_shift
    y4_shifted = y4_rotated + y_shift

    # flip
    if flip_vertically:
        y1_final = 2 * center_y - y1_shifted
        y2_final = 2 * center_y - y2_shifted
        y3_final = 2 * center_y - y3_shifted
        y4_final = 2 * center_y - y4_shifted
    else:
        y1_final = y1_shifted
        y2_final = y2_shifted
        y3_final = y3_shifted
        y4_final = y4_shifted

    if flip_horizontally:
        x1_final = 2 * center_x - x1_shifted
        x2_final = 2 * center_x - x2_shifted
        x3_final = 2 * center_x - x3_shifted
        x4_final = 2 * center_x - x4_shifted
    else:
        x1_final = x1_shifted
        x2_final = x2_shifted
        x3_final = x3_shifted
        x4_final = x4_shifted

    x1_final = round(x1_final, 2)
    y1_final = round(y1_final, 2)
    x2_final = round(x2_final, 2)
    y2_final = round(y2_final, 2)
    x3_final = round(x3_final, 2)
    y3_final = round(y3_final, 2)
    x4_final = round(x4_final, 2)
    y4_final = round(y4_final, 2)

    return [(x1_final, y1_final), (x2_final, y2_final), (x3_final, y3_final), (x4_final, y4_final)]


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
    if location.startswith("http") or location.startswith("https"):
        response = r.get(location)
        image = Image.open(BytesIO(response.content))
    else:
        if not os.path.exists(location):
            raise FileNotFoundError(f"File not found: {location}")
        image = Image.open(location)

    # Rotate the image by the specified degree
    rotated_image = image.rotate(-degree, resample=Image.BICUBIC, expand=False)

    return rotated_image


def save_uploaded_image(
    image: Union[Image.Image, str, np.ndarray, io.BytesIO, InternalFileObject],
    project_uuid,
    frame_uuid=None,
    stage_type=None,
):
    """
    saves the image file (which can be a PIL, arr, InternalFileObject or url) into the project, without
    any tags or logs. then adds that file as the source_image/primary_image, depending
    on the stage selected
    """
    data_repo = DataRepo()

    try:
        if isinstance(image, InternalFileObject):
            saved_image = image
        else:
            saved_image = save_new_image(image, project_uuid)

        # Update records based on stage_type
        if stage_type == WorkflowStageType.SOURCE.value:
            data_repo.update_specific_timing(
                frame_uuid, source_image_id=saved_image.uuid, update_in_place=True
            )
        elif stage_type == WorkflowStageType.STYLED.value:
            number_of_image_variants = add_image_variant(saved_image.uuid, frame_uuid)
            promote_image_variant(frame_uuid, number_of_image_variants - 1)

        return saved_image
    except Exception as e:
        print(f"Failed to save image file due to: {str(e)}")
        return None


# TODO: change variant_to_promote_frame_number to variant_uuid
def promote_image_variant(timing_uuid, variant_to_promote_frame_number: str):
    """
    this methods promotes the variant to the primary image (also referred to as styled image)
    interpolated_clips/videos of the shot are not cleared
    """
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    # promoting variant
    variant_to_promote = timing.alternative_images_list[variant_to_promote_frame_number]
    data_repo.update_specific_timing(
        timing_uuid, primary_image_id=variant_to_promote.uuid
    )  # removing the update_in_place arg for now
    _ = data_repo.get_timing_list_from_shot(timing.shot.uuid)


def promote_video_variant(shot_uuid, variant_uuid):
    """
    this first changes the duration of the interpolated_clip to the frame clip_duration
    then adds the clip to the timed_clip (which is considered as the main variant)
    """
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    variant_to_promote = None
    for variant in shot.interpolated_clip_list:
        if variant.uuid == variant_uuid:
            variant_to_promote = variant
            break

    if not variant_to_promote:
        return None

    # NOTE: removing speed udpate for now
    # if variant_to_promote.location.startswith(('http://', 'https://')):
    #     temp_video_path, _ = urllib3.request.urlretrieve(variant_to_promote.location)
    #     video = VideoFileClip(temp_video_path)
    # else:
    #     video = VideoFileClip(variant_to_promote.location)

    # if video.duration != shot.duration:
    #     video_bytes = VideoProcessor.update_video_speed(
    #         variant_to_promote.location,
    #         shot.duration
    #     )

    #     hosted_url = save_or_host_file_bytes(video_bytes, variant_to_promote.local_path)
    #     if hosted_url:
    #         data_repo.update_file(video.uuid, hosted_url=hosted_url)

    data_repo.update_shot(uuid=shot.uuid, main_clip_id=variant_to_promote.uuid)


def get_canny_img(img_obj, low_threshold, high_threshold, invert_img=False):
    if isinstance(img_obj, str):
        if img_obj.startswith("http"):
            response = r.get(img_obj)
            image_data = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(img_obj, cv2.IMREAD_GRAYSCALE)
    else:
        image_data = generate_pil_image(img_obj)
        image = np.array(image_data)

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    inverted_canny_edges = 255 - canny_edges if invert_img else canny_edges
    new_canny_image = Image.fromarray(inverted_canny_edges)
    return new_canny_image


def extract_canny_lines(
    image_path_or_url, project_uuid, low_threshold=50, high_threshold=150
) -> InternalFileObject:
    data_repo = DataRepo()
    new_canny_image = get_canny_img(image_path_or_url, low_threshold, high_threshold)

    # Save the new image
    unique_file_name = str(uuid.uuid4()) + ".png"
    file_path = f"videos/{project_uuid}/assets/resources/masks/{unique_file_name}"
    hosted_url = save_or_host_file(new_canny_image, file_path)

    file_data = {"name": unique_file_name, "type": InternalFileType.IMAGE.value, "project_id": project_uuid}

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": file_path})

    canny_image_file = data_repo.create_file(**file_data)
    return canny_image_file


def combine_mask_and_input_image(mask_path, input_image_path, overlap_color="transparent"):
    # Open the input image and the mask
    input_image = (
        Image.open(input_image_path) if not isinstance(input_image_path, Image.Image) else input_image_path
    )
    mask_image = Image.open(mask_path) if not isinstance(mask_path, Image.Image) else mask_path
    input_image = input_image.convert("RGBA")

    # Resize mask to match input_image dimensions if they differ
    if mask_image.size != input_image.size:
        mask_image = mask_image.resize(input_image.size, Image.LANCZOS)

    # Convert mask to RGBA if it's not already
    if mask_image.mode != 'RGBA':
        mask_image = mask_image.convert('RGBA')

    is_white = lambda pixel, threshold=245: all(value > threshold for value in pixel[:3])
    fill_color = (128, 128, 128, 255)  # default grey
    if overlap_color == "transparent":
        fill_color = (0, 0, 0, 0)
    elif overlap_color == "grey":
        fill_color = (128, 128, 128, 255)

    for x in range(input_image.width):
        for y in range(input_image.height):
            if is_white(mask_image.getpixel((x, y))):
                input_image.putpixel((x, y), fill_color)

    return input_image


# the input image is an image created by the PIL library
def create_or_update_mask(timing_uuid, image) -> InternalFileObject:
    data_repo = DataRepo()
    timing = data_repo.get_timing_from_uuid(timing_uuid)

    unique_file_name = str(uuid.uuid4()) + ".png"
    file_location = f"videos/{timing.shot.project.uuid}/assets/resources/masks/{unique_file_name}"

    hosted_url = save_or_host_file(image, file_location)
    # if mask is not present than creating a new one
    if not (timing.mask and timing.mask.location):
        file_data = {"name": unique_file_name, "type": InternalFileType.IMAGE.value}

        if hosted_url:
            file_data.update({"hosted_url": hosted_url})
        else:
            file_data.update({"local_path": file_location})

        mask_file: InternalFileObject = data_repo.create_file(**file_data)
        data_repo.update_specific_timing(timing_uuid, mask_id=mask_file.uuid, update_in_place=True)
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

    shot_data = {"project_uuid": project_uuid, "desc": "", "name": name, "duration": 10}

    shot = data_repo.create_shot(**shot_data)
    return shot


# adds the image file in variant (alternative images) list
def add_image_variant(image_file_uuid: str, timing_uuid: str):
    data_repo = DataRepo()
    image_file: InternalFileObject = data_repo.get_file_from_uuid(image_file_uuid)
    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)

    alternative_image_list = timing.alternative_images_list + [image_file]
    alternative_image_uuid_list = [img.uuid for img in alternative_image_list]
    primary_image_uuid = alternative_image_uuid_list[0]
    alternative_image_uuid_list = json.dumps(alternative_image_uuid_list)

    data_repo.update_specific_timing(
        timing_uuid, alternative_images=alternative_image_uuid_list, update_in_place=True
    )

    if not timing.primary_image:
        data_repo.update_specific_timing(
            timing_uuid, primary_image_id=primary_image_uuid, update_in_place=True
        )

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
            data["hosted_url"] = hosted_url
        else:
            data["local_path"] = file_path

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
    file_data = {"name": filename, "type": InternalFileType.IMAGE.value, "project_id": project_uuid}

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": background_img_path})

    image_file = data_repo.create_file(**file_data)

    return image_file


# TODO: don't save or upload image where just passing the PIL object can work
def resize_image(video_name, new_width, new_height, image_file: InternalFileObject) -> InternalFileObject:
    if "http" in image_file.location:
        response = r.get(image_file.location)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file.location)
    resized_image = image.resize((new_width, new_height))

    time.sleep(0.1)

    unique_id = str(uuid.uuid4())
    filepath = "videos/" + str(video_name) + "/temp_image-" + unique_id + ".png"

    hosted_url = save_or_host_file(resized_image, filepath)
    file_data = {"name": str(uuid.uuid4()) + ".png", "type": InternalFileType.IMAGE.value}

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": filepath})

    data_repo = DataRepo()
    image_file = data_repo.create_file(**file_data)

    return image_file


def get_audio_bytes_for_slice(timing_uuid):
    data_repo = DataRepo()

    timing: InternalFrameTimingObject = data_repo.get_timing_from_uuid(timing_uuid)
    project_settings: InternalSettingObject = data_repo.get_project_setting(timing.shot.project.uuid)

    # TODO: add null check for the audio
    audio = AudioSegment.from_file(project_settings.audio.local_path)

    # DOUBT: is it checked if it is the last frame or not?
    audio = audio[timing.frame_time * 1000 : data_repo.get_next_timing(timing_uuid)["frame_time"] * 1000]
    audio_bytes = BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)
    return audio_bytes


def create_frame_inside_shot(shot_uuid, aux_frame_index):
    data_repo = DataRepo()

    timing_data = {
        "shot_id": shot_uuid,
        "animation_style": AnimationStyleType.CREATIVE_INTERPOLATION.value,
        "aux_frame_index": aux_frame_index,
    }
    timing: InternalFrameTimingObject = data_repo.create_timing(**timing_data)

    return timing


def save_audio_file(uploaded_file, project_uuid):
    data_repo = DataRepo()

    local_file_location = os.path.join(f"videos/{project_uuid}/assets/resources/audio", uploaded_file.name)

    audio_bytes = uploaded_file.read()
    hosted_url = save_or_host_file_bytes(audio_bytes, local_file_location, ".mp3")

    file_data = {
        "name": str(uuid.uuid4()) + ".mp3",
        "type": InternalFileType.AUDIO.value,
        "project_id": project_uuid,
    }

    if hosted_url:
        file_data.update({"hosted_url": hosted_url})
    else:
        file_data.update({"local_path": local_file_location})

    audio_file: InternalFileObject = data_repo.create_file(**file_data)
    data_repo.update_project_setting(project_uuid, audio_id=audio_file.uuid)

    return audio_file


# if the output is present it adds it to the respective place or else it updates the inference log
# NOTE: every function used in this should not change/modify session state in anyway
def process_inference_output(**kwargs):
    data_repo = DataRepo()

    inference_time = 0.0
    inference_type = kwargs.get("inference_type")
    log_uuid = None
    # ------------------- FRAME TIMING IMAGE INFERENCE -------------------
    if inference_type == InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value:
        output = kwargs.get("output")
        if output:
            timing_uuid = kwargs.get("timing_uuid")
            promote_new_generation = kwargs.get("promote_new_generation")

            timing = data_repo.get_timing_from_uuid(timing_uuid)
            if not timing:
                return False

            filename = str(uuid.uuid4()) + ".png"
            log_uuid = kwargs.get("log_uuid")
            log = data_repo.get_inference_log_from_uuid(log_uuid)
            if log and log.total_inference_time:
                inference_time = log.total_inference_time

            output_file = data_repo.create_file(
                name=filename,
                type=InternalFileType.IMAGE.value,
                hosted_url=output[0] if isinstance(output, list) else output,
                inference_log_id=log.uuid,
                project_id=timing.shot.project.uuid,
                shot_uuid=kwargs["shot_uuid"] if "shot_uuid" in kwargs else "",
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
            log_uuid = kwargs.get("log_uuid")
            del kwargs["log_uuid"]
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    # --------------------- MULTI VIDEO INFERENCE (INTERPOLATION + MORPHING) -------------------
    elif inference_type == InferenceType.FRAME_INTERPOLATION.value:
        output = kwargs.get("output")
        log_uuid = kwargs.get("log_uuid")

        if output:
            settings = kwargs.get("settings")
            shot_uuid = kwargs.get("shot_uuid")
            shot = data_repo.get_shot_from_uuid(shot_uuid)
            if not shot:
                return False

            output = output[-1] if isinstance(output, list) else output
            # output can also be an url
            if isinstance(output, str):
                if output.startswith("http"):
                    temp_output_file = generate_temp_file(output, ".mp4")
                    output = None
                    with open(temp_output_file.name, "rb") as f:
                        output = f.read()

                    os.remove(temp_output_file.name)
                else:
                    with open(output, "rb") as f:
                        output = f.read()

            # if 'normalise_speed' in settings and settings['normalise_speed']:
            #     output = VideoProcessor.update_video_bytes_speed(output, shot.duration)

            video_location = (
                "videos/" + str(shot.project.uuid) + "/assets/videos/completed/" + str(uuid.uuid4()) + ".mp4"
            )
            video = convert_bytes_to_file(
                file_location_to_save=video_location,
                mime_type="video/mp4",
                file_bytes=output,
                project_uuid=shot.project.uuid,
                inference_log_id=log_uuid,
            )

            if not shot.main_clip or settings.get("promote_to_main_variant", False):
                output_video = sync_audio_and_duration(video, shot_uuid)
                data_repo.update_shot(uuid=shot_uuid, main_clip_id=output_video.uuid)
                data_repo.add_interpolated_clip(shot_uuid, interpolated_clip_id=output_video.uuid)
            else:
                data_repo.add_interpolated_clip(shot_uuid, interpolated_clip_id=video.uuid)

            log = data_repo.get_inference_log_from_uuid(log_uuid)
            if log and log.total_inference_time:
                inference_time = log.total_inference_time
        else:
            del kwargs["log_uuid"]
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    # --------------------- GALLERY IMAGE GENERATION ------------------------
    elif inference_type == InferenceType.GALLERY_IMAGE_GENERATION.value:
        output = kwargs.get("output")

        if output:
            log_uuid = kwargs.get("log_uuid")
            project_uuid = kwargs.get("project_uuid")
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
                tag=InternalFileTag.TEMP_GALLERY_IMAGE.value,  # will be updated to GALLERY_IMAGE once the user clicks 'check for new images'
                shot_uuid=kwargs["shot_uuid"] if "shot_uuid" in kwargs else "",
            )
        else:
            log_uuid = kwargs.get("log_uuid")
            del kwargs["log_uuid"]
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    # --------------------- FRAME INPAINTING ------------------------
    elif inference_type == InferenceType.FRAME_INPAINTING.value:
        output = kwargs.get("output")
        log_uuid = kwargs.get("log_uuid")

        if output:
            stage = kwargs.get("stage", WorkflowStageType.STYLED.value)
            promote = kwargs.get("promote_generation", False)
            current_frame_uuid = kwargs.get("timing_uuid")
            timing = data_repo.get_timing_from_uuid(current_frame_uuid)

            file_name = str(uuid.uuid4()) + ".png"
            output_file = data_repo.create_file(
                name=file_name,
                type=InternalFileType.IMAGE.value,
                hosted_url=output[0] if isinstance(output, list) else output,
                inference_log_id=str(log_uuid),
                project_id=timing.shot.project.uuid,
            )

            if stage == WorkflowStageType.SOURCE.value:
                data_repo.update_specific_timing(
                    current_frame_uuid, source_image_id=output_file.uuid, update_in_place=True
                )
            elif stage == WorkflowStageType.STYLED.value:
                number_of_image_variants = add_image_variant(output_file.uuid, current_frame_uuid)
                if promote:
                    promote_image_variant(current_frame_uuid, number_of_image_variants - 1)

            log = data_repo.get_inference_log_from_uuid(log_uuid)
            if log and log.total_inference_time:
                inference_time = log.total_inference_time
        else:
            del kwargs["log_uuid"]
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    # --------------------- MOTION LORA TRAINING --------------------------
    elif inference_type == InferenceType.MOTION_LORA_TRAINING.value:
        output = kwargs.get("output")
        log_uuid = kwargs.get("log_uuid")

        if output and len(output):
            # output is a list of generated videos
            # we store video_url <--> motion_lora map in a json file

            # NOTE: need to convert 'lora_trainer' into a separate module if it needs to work on hosted version
            # fetching the current generated loras
            spatial_lora_path = os.path.join(COMFY_BASE_PATH, "models", "loras", "trained_spatial")
            temporal_lora_path = os.path.join(COMFY_BASE_PATH, "models", "animatediff_motion_lora")
            lora_path = temporal_lora_path
            _, latest_trained_files = get_latest_project_files(lora_path)

            cur_idx, data = 0, {}
            for vid in output:
                if vid.endswith(".gif"):
                    data[latest_trained_files[cur_idx]] = vid
                    cur_idx += 1

            write_to_motion_lora_local_db(data)
        else:
            del kwargs["log_uuid"]
            data_repo.update_inference_log_origin_data(log_uuid, **kwargs)

    if inference_time:
        credits_used = round(inference_time * 0.004, 3)  # make this more granular for different models
        data_repo.update_usage_credits(-credits_used, log_uuid)

    return True


def get_latest_project_files(parent_directory):
    latest_project = None
    latest_time = 0

    for date_folder in os.listdir(parent_directory):
        date_folder_path = os.path.join(parent_directory, date_folder)

        if os.path.isdir(date_folder_path):
            for time_folder in os.listdir(date_folder_path):
                time_folder_path = os.path.join(date_folder_path, time_folder)

                if os.path.isdir(time_folder_path):
                    for project_name_folder in os.listdir(time_folder_path):
                        project_folder_path = os.path.join(time_folder_path, project_name_folder)

                        if os.path.isdir(project_folder_path):
                            creation_time = os.path.getctime(project_folder_path)

                            if creation_time > latest_time:
                                latest_time = creation_time
                                latest_project = project_folder_path

    if latest_project:
        latest_files = sorted(os.listdir(latest_project))
        return latest_project, latest_files
    else:
        return None, None


def check_project_meta_data(project_uuid):
    """
    checking for project metadata (like cache updates - we update specific entities using this flag)
    project_update_data is of the format {"data_update": [timing_uuid], "gallery_update": True/False, "background_img_list": []}
    """
    data_repo = DataRepo()

    with sqlite_atomic_transaction():
        project: InternalProjectObject = data_repo.get_project_from_uuid(project_uuid)
        timing_update_data = (
            json.loads(project.meta_data).get(ProjectMetaData.DATA_UPDATE.value, None)
            if project.meta_data
            else None
        )
        if timing_update_data and len(timing_update_data):
            for timing_uuid in timing_update_data:
                _ = data_repo.get_timing_from_uuid(timing_uuid, invalidate_cache=True)

        gallery_update_data = (
            json.loads(project.meta_data).get(ProjectMetaData.GALLERY_UPDATE.value, False)
            if project.meta_data
            else False
        )
        if gallery_update_data:
            pass

        shot_update_data = (
            json.loads(project.meta_data).get(ProjectMetaData.SHOT_VIDEO_UPDATE.value, [])
            if project.meta_data
            else []
        )
        if shot_update_data and len(shot_update_data):
            for shot_uuid in shot_update_data:
                _ = data_repo.get_shot_list(shot_uuid, invalidate_cache=True)

        # clearing update data from cache
        blank_data_obj = {
            ProjectMetaData.DATA_UPDATE.value: [],
            ProjectMetaData.GALLERY_UPDATE.value: False,
            ProjectMetaData.SHOT_VIDEO_UPDATE.value: [],
        }
        meta_data = json.loads(project.meta_data) if project.meta_data else {}
        meta_data.update(blank_data_obj)
        data_repo.update_project(uuid=project.uuid, meta_data=json.dumps(meta_data))


def update_app_setting_keys():
    # TODO: not in use atm
    # data_repo = DataRepo()
    # app_logger = AppLogger()

    # if OFFLINE_MODE:
    #     key = os.getenv("REPLICATE_KEY", "")
    # else:
    #     import boto3

    #     ssm = boto3.client("ssm", region_name="ap-south-1")
    #     key = ssm.get_parameter(Name="/backend/banodoco/replicate/key")["Parameter"]["Value"]

    # app_setting = data_repo.get_app_secrets_from_user_uuid()
    # if app_setting and app_setting["replicate_key"] == key:
    #     return

    # app_logger.log(LoggingType.DEBUG, "setting keys", None)
    # data_repo.update_app_setting(replicate_username="update")
    # data_repo.update_app_setting(replicate_key=key)
    pass


def random_seed():
    return random.randint(10**14, 10**15 - 1)


# setting up multiprocessing queue for log termination
# NOTE: for some reason concurrent future is not working properly with streamlit (it's not able to pickle methods on state refresh)
def stop_gen(log):
    data_repo = DataRepo()

    in_progress = log.status == InferenceStatus.IN_PROGRESS.value
    data_repo.update_inference_log(uuid=log.uuid, status=InferenceStatus.CANCELED.value)
    print(f"DB update {log.uuid} ----------")

    if in_progress:
        sys.path.append(str(os.getcwd()) + COMFY_RUNNER_PATH[1:])
        from comfy_runner.inf import ComfyRunner

        comfy_runner = ComfyRunner()
        comfy_runner.stop_current_generation(log.uuid, 3)
        print(f"Process stopped {log.uuid} ----------")


def worker(queue):
    while True:
        log = queue.get()
        if log is None:
            break
        stop_gen(log)

def stop_generations_worker():
    queue = multiprocessing.Queue()
    num_workers = 10

    workers = [multiprocessing.Process(target=worker, args=(queue,)) for _ in range(num_workers)]
    for w in workers:
        w.start()

    return queue, workers


def stop_generations(logs: List[InferenceLogObject]):
    queue, workers = stop_generations_worker()

    for log in logs:
        if log.status in [InferenceStatus.IN_PROGRESS.value, InferenceStatus.QUEUED.value]:
            queue.put(log)

    for _ in range(len(workers)):
        queue.put(None)

    for w in workers:
        w.join()
