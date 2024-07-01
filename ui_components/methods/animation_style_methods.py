import json
import os
from typing import List
import streamlit as st
from backend.models import InternalFileObject
from shared.constants import COMFY_BASE_PATH, InferenceParamType, ProjectMetaData
from ui_components.constants import DEFAULT_SHOT_MOTION_VALUES, ShotMetaData
from ui_components.models import InternalProjectObject, InternalShotObject
from utils.common_utils import acquire_lock, release_lock
from utils.data_repo.data_repo import DataRepo
import numpy as np
import matplotlib.pyplot as plt


def get_generation_settings_from_log(log_uuid=None):
    data_repo = DataRepo()
    log = data_repo.get_inference_log_from_uuid(log_uuid)
    input_params = json.loads(log.input_params) if log.input_params else {}
    query_obj = json.loads(input_params.get(InferenceParamType.QUERY_DICT.value, json.dumps({})))
    shot_meta_data = query_obj["data"].get("data", {}).get("shot_data", {})
    data_type = None
    if shot_meta_data and ShotMetaData.MOTION_DATA.value in shot_meta_data:
        data_type = ShotMetaData.MOTION_DATA.value
    elif shot_meta_data and ShotMetaData.DYNAMICRAFTER_DATA.value in shot_meta_data:
        data_type = ShotMetaData.DYNAMICRAFTER_DATA.value

    shot_meta_data = (json.loads(shot_meta_data.get(data_type))) if data_type else None

    return shot_meta_data, data_type


def load_shot_settings(shot_uuid, log_uuid=None, load_images=True, load_setting_values=True):
    data_repo = DataRepo()
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)

    """
    NOTE: every shot's meta_data has the latest copy of the settings (whatever is the most recent gen)
    apart from this, every generation log also has it's own copy of settings (for that particular gen)
    by default shot's settings is applied whenever a new generation is to be created, but if a user
    clicks "load settings" on a particular gen then it's settings are loaded from it's generation log
    """

    """
    NOTE: the logic has been updated and instead of picking the default values from the given shot (shot_uuid)
    the values would be picked from the active_shot, present inside project's meta_data. older code may still
    be present at some places.
    """

    # loading settings of the last generation (saved in the shot)
    # in case no log_uuid is provided
    if not log_uuid:
        shot_meta_data = shot.meta_data_dict.get(ShotMetaData.MOTION_DATA.value, None)
        # if the current shot is newly created and has no meta data
        if not shot_meta_data:
            project_meta_data = json.loads(shot.project.meta_data) if shot.project.meta_data else {}
            active_shot_uuid = project_meta_data.get(ProjectMetaData.ACTIVE_SHOT.value, None)
            if active_shot_uuid:
                active_shot: InternalShotObject = data_repo.get_shot_from_uuid(active_shot_uuid)
                if active_shot:
                    shot_meta_data = active_shot.meta_data_dict.get(ShotMetaData.MOTION_DATA.value, None)
                else:
                    # if shot was deleted then setting the first shot as the active shot
                    shot_list: List[InternalShotObject] = data_repo.get_shot_list(shot.project.uuid)
                    shot_meta_data = shot_list[0].meta_data_dict.get(ShotMetaData.MOTION_DATA.value, None)
                    update_active_shot(shot_list[0].uuid)

        else:
            update_active_shot(shot.uuid)

        shot_meta_data = json.loads(shot_meta_data) if shot_meta_data else {}
        data_type = None
        st.session_state[f"{shot_uuid}_selected_variant_log_uuid"] = None

    # loading settings from that particular log
    else:
        shot_meta_data, data_type = get_generation_settings_from_log(log_uuid)
        if load_images:
            st.session_state[f"{shot_uuid}_selected_variant_log_uuid"] = log_uuid

    if load_setting_values:
        if shot_meta_data:
            if not data_type or data_type == ShotMetaData.MOTION_DATA.value:
                st.session_state[f"type_of_animation_{shot.uuid}"] = 0
                # ------------------ updating timing data
                timing_data = shot_meta_data.get("timing_data", [])
                for idx, _ in enumerate(
                    shot.timing_list
                ):  # fix: check how the image list is being stored here and use that instead
                    # setting default parameters (fetching data from the shot if it's present)
                    if timing_data and len(timing_data) >= idx + 1:
                        motion_data = timing_data[idx]

                    for k, v in motion_data.items():
                        st.session_state[f"{k}_{shot_uuid}_{idx}"] = v

                # --------------------- updating other settings main settings
                main_setting_data = shot_meta_data.get("main_setting_data", {})
                for key in main_setting_data:
                    # if data is being loaded from a different shot then key will have to be updated
                    # from "lora_data_{other_shot_uuid}" to "lora_data_{this_shot_data}"
                    if str(shot_uuid) not in key:
                        new_key = key.rsplit("_", 1)[0] + "_" + str(shot_uuid)
                    else:
                        new_key = key

                    st.session_state[new_key] = main_setting_data[key]
                    if (
                        key == f"structure_control_image_uuid_{shot_uuid}" and not main_setting_data[key]
                    ):  # hackish sol, will fix later
                        st.session_state[f"structure_control_image_{shot_uuid}"] = None

                st.rerun()
            elif data_type == ShotMetaData.DYNAMICRAFTER_DATA.value:
                st.session_state[f"type_of_animation_{shot.uuid}"] = 1
                main_setting_data = shot_meta_data.get("main_setting_data", {})
                for key in main_setting_data:
                    st.session_state[key] = main_setting_data[key]
                st.rerun()
        else:
            for idx, _ in enumerate(shot.timing_list):  # fix: check how the image list is being stored here
                for k, v in DEFAULT_SHOT_MOTION_VALUES.items():
                    st.session_state[f"{k}_{shot_uuid}_{idx}"] = v


def format_frame_prompts_with_buffer(frame_numbers, individual_prompts, buffer):
    adjusted_frame_numbers = [frame + buffer for frame in frame_numbers]

    # Preprocess prompts to remove any '/' or '"' from the values
    processed_prompts = [prompt.replace("/", "").replace('"', "") for prompt in individual_prompts]

    # Format the adjusted frame numbers and processed prompts
    formatted = ", ".join(
        f'"{int(frame)}": "{prompt}"' for frame, prompt in zip(adjusted_frame_numbers, processed_prompts)
    )
    return formatted


def plot_weights(weights_list, frame_numbers_list):
    plt.figure(figsize=(12, 6))
    for i, weights in enumerate(weights_list):
        frame_numbers = [frame_number / 100 for frame_number in frame_numbers_list[i]]
        plt.plot(frame_numbers, weights, label=f"Frame {i + 1}")

    # Plot settings
    plt.xlabel("Seconds")
    plt.ylabel("Weight")
    plt.legend()
    plt.ylim(0, 1.0)
    plt.show()
    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.pyplot()


def calculate_weights(
    keyframe_positions, strength_values, buffer, key_frame_influence_values, last_key_frame_position
):
    def calculate_influence_frame_number(key_frame_position, next_key_frame_position, distance):
        # Calculate the absolute distance between key frames
        key_frame_distance = abs(next_key_frame_position - key_frame_position)

        # Apply the distance multiplier
        extended_distance = key_frame_distance * distance

        # Determine the direction of influence based on the positions of the key frames
        if key_frame_position < next_key_frame_position:
            # Normal case: influence extends forward
            influence_frame_number = key_frame_position + extended_distance
        else:
            # Reverse case: influence extends backward
            influence_frame_number = key_frame_position - extended_distance

        # Return the result rounded to the nearest integer
        return round(influence_frame_number)

    def find_curve(
        batch_index_from,
        batch_index_to,
        strength_from,
        strength_to,
        interpolation,
        revert_direction_at_midpoint,
        last_key_frame_position,
        i,
        number_of_items,
        buffer,
    ):
        # Initialize variables based on the position of the keyframe
        range_start = batch_index_from
        range_end = batch_index_to
        # if it's the first value, set influence range from 1.0 to 0.0
        if i == number_of_items - 1:
            range_end = last_key_frame_position

        steps = range_end - range_start
        diff = strength_to - strength_from

        # Calculate index for interpolation
        index = (
            np.linspace(0, 1, steps // 2 + 1) if revert_direction_at_midpoint else np.linspace(0, 1, steps)
        )

        # Calculate weights based on interpolation type
        if interpolation == "linear":
            weights = np.linspace(strength_from, strength_to, len(index))
        elif interpolation == "ease-in":
            weights = diff * np.power(index, 2) + strength_from
        elif interpolation == "ease-out":
            weights = diff * (1 - np.power(1 - index, 2)) + strength_from
        elif interpolation == "ease-in-out":
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + strength_from

        if revert_direction_at_midpoint:
            weights = np.concatenate([weights, weights[::-1]])

        # Generate frame numbers
        frame_numbers = np.arange(range_start, range_start + len(weights))

        # "Dropper" component: For keyframes with negative start, drop the weights
        if range_start < 0 and i > 0:
            drop_count = abs(range_start)
            weights = weights[drop_count:]
            frame_numbers = frame_numbers[drop_count:]

        # Dropper component: for keyframes a range_End is greater than last_key_frame_position, drop the weights
        if range_end > last_key_frame_position and i < number_of_items - 1:
            drop_count = range_end - last_key_frame_position
            weights = weights[:-drop_count]
            frame_numbers = frame_numbers[:-drop_count]

        return weights, frame_numbers

    weights_list = []
    frame_numbers_list = []

    for i in range(len(keyframe_positions)):
        keyframe_position = keyframe_positions[i]
        interpolation = "ease-in-out"
        # strength_from = strength_to = 1.0

        if i == 0:  # first image
            # GET IMAGE AND KEYFRAME INFLUENCE VALUES
            key_frame_influence_from, key_frame_influence_to = key_frame_influence_values[i]
            start_strength, mid_strength, end_strength = strength_values[i]
            keyframe_position = keyframe_positions[i]
            next_key_frame_position = keyframe_positions[i + 1]
            batch_index_from = keyframe_position
            batch_index_to_excl = calculate_influence_frame_number(
                keyframe_position, next_key_frame_position, key_frame_influence_to
            )
            weights, frame_numbers = find_curve(
                batch_index_from,
                batch_index_to_excl,
                mid_strength,
                end_strength,
                interpolation,
                False,
                last_key_frame_position,
                i,
                len(keyframe_positions),
                buffer,
            )
            # interpolation = "ease-in"

        elif i == len(keyframe_positions) - 1:  # last image
            # GET IMAGE AND KEYFRAME INFLUENCE VALUES
            key_frame_influence_from, key_frame_influence_to = key_frame_influence_values[i]
            start_strength, mid_strength, end_strength = strength_values[i]
            # strength_from, strength_to = cn_strength_values[i-1]
            keyframe_position = keyframe_positions[i]
            previous_key_frame_position = keyframe_positions[i - 1]
            batch_index_from = calculate_influence_frame_number(
                keyframe_position, previous_key_frame_position, key_frame_influence_from
            )
            batch_index_to_excl = keyframe_position
            weights, frame_numbers = find_curve(
                batch_index_from,
                batch_index_to_excl,
                start_strength,
                mid_strength,
                interpolation,
                False,
                last_key_frame_position,
                i,
                len(keyframe_positions),
                buffer,
            )
            # interpolation =  "ease-out"

        else:  # middle images
            # GET IMAGE AND KEYFRAME INFLUENCE VALUES
            key_frame_influence_from, key_frame_influence_to = key_frame_influence_values[i]
            start_strength, mid_strength, end_strength = strength_values[i]
            keyframe_position = keyframe_positions[i]

            # CALCULATE WEIGHTS FOR FIRST HALF
            previous_key_frame_position = keyframe_positions[i - 1]
            batch_index_from = calculate_influence_frame_number(
                keyframe_position, previous_key_frame_position, key_frame_influence_from
            )
            batch_index_to_excl = keyframe_position
            first_half_weights, first_half_frame_numbers = find_curve(
                batch_index_from,
                batch_index_to_excl,
                start_strength,
                mid_strength,
                interpolation,
                False,
                last_key_frame_position,
                i,
                len(keyframe_positions),
                buffer,
            )

            # CALCULATE WEIGHTS FOR SECOND HALF
            next_key_frame_position = keyframe_positions[i + 1]
            batch_index_from = keyframe_position
            batch_index_to_excl = calculate_influence_frame_number(
                keyframe_position, next_key_frame_position, key_frame_influence_to
            )
            second_half_weights, second_half_frame_numbers = find_curve(
                batch_index_from,
                batch_index_to_excl,
                mid_strength,
                end_strength,
                interpolation,
                False,
                last_key_frame_position,
                i,
                len(keyframe_positions),
                buffer,
            )

            # COMBINE FIRST AND SECOND HALF
            weights = np.concatenate([first_half_weights, second_half_weights])
            frame_numbers = np.concatenate([first_half_frame_numbers, second_half_frame_numbers])

        weights_list.append(weights)
        frame_numbers_list.append(frame_numbers)

    return weights_list, frame_numbers_list


def extract_influence_values(
    type_of_key_frame_influence,
    dynamic_key_frame_influence_values,
    keyframe_positions,
    linear_key_frame_influence_value,
):
    # Check and convert linear_key_frame_influence_value if it's a float or string float
    # if it's a string that starts with a parenthesis, convert it to a tuple
    if isinstance(linear_key_frame_influence_value, str) and linear_key_frame_influence_value[0] == "(":
        linear_key_frame_influence_value = eval(linear_key_frame_influence_value)

    if not isinstance(linear_key_frame_influence_value, tuple):
        if isinstance(linear_key_frame_influence_value, (float, str)):
            try:
                value = float(linear_key_frame_influence_value)
                linear_key_frame_influence_value = (value, value)
            except ValueError:
                raise ValueError(
                    "linear_key_frame_influence_value must be a float or a string representing a float"
                )

    number_of_outputs = len(keyframe_positions)
    if type_of_key_frame_influence == "dynamic":
        # Convert list of individual float values into tuples
        if all(isinstance(x, float) for x in dynamic_key_frame_influence_values):
            dynamic_values = [(value, value) for value in dynamic_key_frame_influence_values]
        elif (
            isinstance(dynamic_key_frame_influence_values[0], str)
            and dynamic_key_frame_influence_values[0] == "("
        ):
            string_representation = "".join(dynamic_key_frame_influence_values)
            dynamic_values = eval(f"[{string_representation}]")
        else:
            dynamic_values = (
                dynamic_key_frame_influence_values
                if isinstance(dynamic_key_frame_influence_values, list)
                else [dynamic_key_frame_influence_values]
            )
        return dynamic_values[:number_of_outputs]
    else:
        return [linear_key_frame_influence_value for _ in range(number_of_outputs)]


def extract_strength_values(
    type_of_key_frame_influence,
    dynamic_key_frame_influence_values,
    keyframe_positions,
    linear_key_frame_influence_value,
):
    if type_of_key_frame_influence == "dynamic":
        # Process the dynamic_key_frame_influence_values depending on its format
        if isinstance(dynamic_key_frame_influence_values, str):
            dynamic_values = eval(dynamic_key_frame_influence_values)
        else:
            dynamic_values = dynamic_key_frame_influence_values

        # Iterate through the dynamic values and convert tuples with two values to three values
        dynamic_values_corrected = []
        for value in dynamic_values:
            if len(value) == 2:
                value = (value[0], value[1], value[0])
            dynamic_values_corrected.append(value)

        return dynamic_values_corrected
    else:
        # Process for linear or other types
        if len(linear_key_frame_influence_value) == 2:
            linear_key_frame_influence_value = (
                linear_key_frame_influence_value[0],
                linear_key_frame_influence_value[1],
                linear_key_frame_influence_value[0],
            )
        return [linear_key_frame_influence_value for _ in range(len(keyframe_positions) - 1)]


def get_keyframe_positions(
    type_of_frame_distribution, dynamic_frame_distribution_values, images, linear_frame_distribution_value
):
    if type_of_frame_distribution == "dynamic":
        if isinstance(dynamic_frame_distribution_values, str):
            # Sort the keyframe positions in numerical order
            return sorted([int(kf.strip()) for kf in dynamic_frame_distribution_values.split(",")])
        elif isinstance(dynamic_frame_distribution_values, list):
            return sorted(dynamic_frame_distribution_values)
    else:
        # Calculate the number of keyframes based on the total duration and linear_frames_per_keyframe
        return [i * linear_frame_distribution_value for i in range(len(images))]


postfix_str = "_generate_inference"


def toggle_generate_inference(position, **kwargs):

    for k, v in kwargs.items():
        st.session_state[k] = v
    if position + postfix_str not in st.session_state:
        st.session_state[position + postfix_str] = True
    else:
        st.session_state[position + postfix_str] = not st.session_state[position + postfix_str]


def is_inference_enabled(position):
    if f"{position}{postfix_str}" in st.session_state and st.session_state[f"{position}{postfix_str}"]:
        return True
    return False


def transform_data(
    strength_of_frames,
    movements_between_frames,
    speeds_of_transitions,
    distances_to_next_frames,
    type_of_motion_context,
    strength_of_adherence,
    individual_prompts,
    individual_negative_prompts,
    buffer,
    motions_during_frames,
):
    # FRAME SETTINGS
    def adjust_and_invert_relative_value(middle_value, relative_value):
        if relative_value is not None:
            adjusted_value = middle_value * relative_value
            return round(middle_value - adjusted_value, 2)
        return None

    def invert_value(value):
        return round(1.0 - value, 2) if value is not None else None

    # Creating output_strength with relative and inverted start and end values
    output_strength = []
    for i, strength in enumerate(strength_of_frames):
        start_value = None if i == 0 else movements_between_frames[i - 1]
        end_value = None if i == len(strength_of_frames) - 1 else movements_between_frames[i]

        # Adjusting and inverting start and end values relative to the middle value
        adjusted_start = adjust_and_invert_relative_value(strength, start_value)
        adjusted_end = adjust_and_invert_relative_value(strength, end_value)

        output_strength.append((adjusted_start, strength, adjusted_end))

    # Creating output_speeds with inverted values
    output_speeds = [(None, None) for _ in range(len(speeds_of_transitions) + 1)]
    for i in range(len(speeds_of_transitions)):
        current_tuple = list(output_speeds[i])
        next_tuple = list(output_speeds[i + 1])

        inverted_speed = invert_value(speeds_of_transitions[i])
        current_tuple[1] = inverted_speed * 2
        next_tuple[0] = inverted_speed * 2

        output_speeds[i] = tuple(current_tuple)
        output_speeds[i + 1] = tuple(next_tuple)

    # Creating cumulative_distances
    cumulative_distances = [0]
    for distance in distances_to_next_frames:
        cumulative_distances.append(cumulative_distances[-1] + distance)

    cumulative_distances = [int(float(value) * 8) for value in cumulative_distances]

    # MOTION CONTEXT SETTINGS
    if type_of_motion_context == "Low":
        context_length = 16
        context_stride = 1
        context_overlap = 2

    elif type_of_motion_context == "Standard":
        context_length = 16
        context_stride = 2
        context_overlap = 4

    elif type_of_motion_context == "High":
        context_length = 16
        context_stride = 4
        context_overlap = 4

    # SPARSE CTRL SETTINGS
    multipled_base_end_percent = 0.05 * (strength_of_adherence * 10)
    multipled_base_adapter_strength = 0.05 * (strength_of_adherence * 20)

    # FRAME PROMPTS FORMATTING
    def format_frame_prompts_with_buffer(frame_numbers, individual_prompts, buffer):
        adjusted_frame_numbers = [frame + buffer for frame in frame_numbers]

        # Preprocess prompts to remove any '/' or '"' from the values
        processed_prompts = [prompt.replace("/", "").replace('"', "") for prompt in individual_prompts]

        # Format the adjusted frame numbers and processed prompts
        formatted = ", ".join(
            f'"{int(frame)}": "{prompt}"' for frame, prompt in zip(adjusted_frame_numbers, processed_prompts)
        )
        return formatted

    # Applying format_frame_prompts_with_buffer
    formatted_individual_prompts = format_frame_prompts_with_buffer(
        cumulative_distances, individual_prompts, buffer
    )
    formatted_individual_negative_prompts = format_frame_prompts_with_buffer(
        cumulative_distances, individual_negative_prompts, buffer
    )

    # MOTION STRENGTHS FORMATTING
    adjusted_frame_numbers = [0] + [frame + buffer for frame in cumulative_distances[1:]]

    # Format the adjusted frame numbers and strengths
    motions_during_frames = ", ".join(
        f"{int(frame)}:({strength})" for frame, strength in zip(adjusted_frame_numbers, motions_during_frames)
    )

    return (
        output_strength,
        output_speeds,
        cumulative_distances,
        context_length,
        context_stride,
        context_overlap,
        multipled_base_end_percent,
        multipled_base_adapter_strength,
        formatted_individual_prompts,
        formatted_individual_negative_prompts,
        motions_during_frames,
    )


def get_timing_data(
    shot_uuid,
    img_list,
    strength_of_frames,
    distances_to_next_frames,
    speeds_of_transitions,
    freedoms_between_frames,
    motions_during_frames,
    individual_prompts,
    individual_negative_prompts,
):
    timing_data = []
    for idx, img in enumerate(img_list):
        # updating the session state rn
        st.session_state[f"strength_of_frame_{shot_uuid}_{idx}"] = strength_of_frames[idx]
        st.session_state[f"individual_prompt_{shot_uuid}_{idx}"] = individual_prompts[idx]
        st.session_state[f"individual_negative_prompt_{shot_uuid}_{idx}"] = individual_negative_prompts[idx]
        st.session_state[f"motion_during_frame_{shot_uuid}_{idx}"] = motions_during_frames[idx]
        st.session_state[f"distance_to_next_frame_{shot_uuid}_{idx}"] = (
            distances_to_next_frames[idx] if idx < len(img_list) - 1 else distances_to_next_frames[idx - 1]
        )
        st.session_state[f"speed_of_transition_{shot_uuid}_{idx}"] = (
            speeds_of_transitions[idx] if idx < len(img_list) - 1 else speeds_of_transitions[idx - 1]
        )
        st.session_state[f"freedom_between_frames_{shot_uuid}_{idx}"] = (
            freedoms_between_frames[idx] if idx < len(img_list) - 1 else freedoms_between_frames[idx - 1]
        )

        # adding into the meta-data. this is what is finally stored in the shot and inference log
        state_data = {
            "strength_of_frame": strength_of_frames[idx],
            "individual_prompt": individual_prompts[idx],
            "individual_negative_prompt": individual_negative_prompts[idx],
            "motion_during_frame": motions_during_frames[idx],
            "distance_to_next_frame": (
                distances_to_next_frames[idx]
                if idx < len(img_list) - 1
                else distances_to_next_frames[idx - 1]
            ),
            "speed_of_transition": (
                speeds_of_transitions[idx] if idx < len(img_list) - 1 else speeds_of_transitions[idx - 1]
            ),
            "freedom_between_frames": (
                freedoms_between_frames[idx] if idx < len(img_list) - 1 else freedoms_between_frames[idx - 1]
            ),
        }

        timing_data.append(state_data)

    return timing_data


def update_session_state_with_animation_details(
    shot_uuid,
    img_list: List[InternalFileObject],
    strength_of_frames,
    distances_to_next_frames,
    speeds_of_transitions,
    freedoms_between_frames,
    motions_during_frames,
    individual_prompts,
    individual_negative_prompts,
    lora_data,
    default_model,
    high_detail_mode=True,
    structure_control_img_uuid=None,
    strength_of_structure_control_img=None,
    type_of_generation_index=0,
    stabilise_motion=None
):
    """
    for any generation session_state holds two kind of data objects.
    1. timing_data -> this is data points like distance to next frames, frame strengths etc.. basically
    anything to do with timing/frames
    2. main_setting_data -> this is the model selected, lora added, workflow selected etc..
    """

    """
    A 'active_shot' index is maintained and settings are picked from that shot, whenvever
    generating a new shot. But when someone wants to save the settings manually,the data is saved
    in the shot (the last generation data is maintained in both the shot and the generation log)
    """
    from utils.constants import StabliseMotionOption
    
    data_repo = DataRepo()

    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
    meta_data = shot.meta_data_dict
    timing_data = get_timing_data(
        shot_uuid,
        img_list,
        strength_of_frames,
        distances_to_next_frames,
        speeds_of_transitions,
        freedoms_between_frames,
        motions_during_frames,
        individual_prompts,
        individual_negative_prompts,
    )

    main_setting_data = {}
    main_setting_data[f"lora_data_{shot.uuid}"] = lora_data
    main_setting_data[f"strength_of_adherence_value_{shot.uuid}"] = st.session_state["strength_of_adherence"]
    main_setting_data[f"type_of_motion_context_index_{shot.uuid}"] = st.session_state[
        "type_of_motion_context"
    ]
    main_setting_data[f"positive_prompt_video_{shot.uuid}"] = st.session_state["overall_positive_prompt"]
    main_setting_data[f"negative_prompt_video_{shot.uuid}"] = st.session_state["overall_negative_prompt"]
    main_setting_data[f"structure_control_image_uuid_{shot.uuid}"] = structure_control_img_uuid
    main_setting_data[f"saved_strength_of_structure_control_image_{shot.uuid}"] = (
        strength_of_structure_control_img
    )
    main_setting_data[f"type_of_generation_index_{shot.uuid}"] = type_of_generation_index
    main_setting_data[f"high_detail_mode_val_{shot.uuid}"] = high_detail_mode
    main_setting_data[f"stabilise_motion_{shot.uuid}"] = stabilise_motion or StabliseMotionOption.NONE.value

    checkpoints_dir = os.path.join(COMFY_BASE_PATH, "models", "checkpoints")
    all_files = os.listdir(checkpoints_dir)
    model_files = [file for file in all_files if file.endswith(".safetensors") or file.endswith(".ckpt")]
    model_files = [file for file in model_files if "xl" not in file]

    if "sd_model_video" in st.session_state and len(model_files):
        idx = (
            model_files.index(st.session_state["sd_model_video"])
            if st.session_state["sd_model_video"] in model_files
            else 0
        )
        main_setting_data[f"ckpt_{shot.uuid}"] = model_files[idx]
    else:
        main_setting_data[f"ckpt_{shot.uuid}"] = default_model

    update_data = {
        ShotMetaData.MOTION_DATA.value: json.dumps(
            {"timing_data": timing_data, "main_setting_data": main_setting_data}
        )
    }

    meta_data.update(update_data)
    data_repo.update_shot(**{"uuid": shot_uuid, "meta_data": json.dumps(meta_data)})
    update_active_shot(shot_uuid)
    return update_data


def update_active_shot(shot_uuid):
    # updating the active shot inside the project
    data_repo = DataRepo()
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)
    key = shot.project.uuid
    if acquire_lock(key):
        project: InternalProjectObject = data_repo.get_project_from_uuid(uuid=key)
        if project:
            meta_data = json.loads(project.meta_data) if project.meta_data else {}
            meta_data[ProjectMetaData.ACTIVE_SHOT.value] = str(shot_uuid)
            data_repo.update_project(uuid=project.uuid, meta_data=json.dumps(meta_data))
        release_lock(key)


# saving dynamic crafter generation details
def update_session_state_with_dc_details(shot_uuid, img_list, video_desc):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    meta_data = shot.meta_data_dict
    main_setting_data = {}
    for idx, img in enumerate(img_list):
        main_setting_data[f"img{idx+1}_uuid_{shot_uuid}"] = img.uuid

    main_setting_data[f"video_desc_{shot_uuid}"] = video_desc
    update_data = {
        ShotMetaData.DYNAMICRAFTER_DATA.value: json.dumps({"main_setting_data": main_setting_data})
    }
    meta_data.update(update_data)
    data_repo.update_shot(**{"uuid": shot_uuid, "meta_data": json.dumps(meta_data)})
    return update_data
