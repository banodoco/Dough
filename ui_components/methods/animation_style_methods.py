import json
import time
import streamlit as st
from shared.constants import InferenceParamType
from ui_components.constants import DEFAULT_SHOT_MOTION_VALUES, ShotMetaData
from utils.data_repo.data_repo import DataRepo


def get_generation_settings_from_log(log_uuid=None):
    data_repo = DataRepo()
    log = data_repo.get_inference_log_from_uuid(log_uuid)
    input_params = json.loads(log.input_params) if log.input_params else {}
    query_obj = json.loads(input_params.get(InferenceParamType.QUERY_DICT.value, json.dumps({})))
    shot_meta_data = query_obj['data'].get('data', {}).get("shot_data", {})
    shot_meta_data = json.loads(shot_meta_data.get("motion_data")) if shot_meta_data \
        and "motion_data" in shot_meta_data else None
    
    return shot_meta_data

def load_shot_settings(shot_uuid, log_uuid=None):
    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)
    
    # loading settings of the last generation (saved in the shot)
    # in case no log_uuid is provided
    if not log_uuid:
        shot_meta_data = shot.meta_data_dict.get(ShotMetaData.MOTION_DATA.value, json.dumps({}))
        shot_meta_data = json.loads(shot_meta_data)
                    
    # loading settings from that particular log
    else:
        shot_meta_data = get_generation_settings_from_log(log_uuid)
    
    if shot_meta_data:
        # updating timing data
        timing_data = shot_meta_data.get("timing_data", [])
        for idx, _ in enumerate(shot.timing_list):
            # setting default parameters (fetching data from the shot if it's present)
            if timing_data and len(timing_data) >= idx + 1:
                motion_data = timing_data[idx]

            for k, v in motion_data.items():
                st.session_state[f'{k}_{shot_uuid}_{idx}'] = v
        
        # updating other settings main settings
        main_setting_data = shot_meta_data.get("main_setting_data", {})
        for key in main_setting_data:
                st.session_state[key] = main_setting_data[key]
    else:
        for idx, _ in enumerate(shot.timing_list):
            for k, v in DEFAULT_SHOT_MOTION_VALUES.items():
                st.session_state[f"{k}_{shot_uuid}_{idx}"] = v
            
def reverse_data_transformation(dynamic_strength_values, dynamic_key_frame_influence_values, dynamic_frame_distribution_values, context_length, context_stride, context_overlap, multipled_base_end_percent, formatted_individual_prompts, formatted_individual_negative_prompts, formatted_motions, buffer):

    def reverse_transform(dynamic_strength_values, dynamic_key_frame_influence_values, dynamic_frame_distribution_values):

        # Reconstructing strength_of_frames
        strength_of_frames = [strength for _, strength, _ in dynamic_strength_values]
        
        # Reconstructing freedoms_between_frames (correctly as movements_between_frames)
        freedoms_between_frames = []
        for i in range(1, len(dynamic_strength_values)):
            if dynamic_strength_values[i][0] is not None:
                middle_value = dynamic_strength_values[i][1]
                adjusted_value = dynamic_strength_values[i][0]
                relative_value = (middle_value - adjusted_value) / middle_value
                freedoms_between_frames.append(round(relative_value, 2))  # Ensure proper rounding
        
        # Reconstructing speeds_of_transitions with correct rounding
        speeds_of_transitions = []
        for current, next_ in dynamic_key_frame_influence_values[:-1]:
            if next_ is not None:
                inverted_speed = next_ / 2
                original_speed = 1.0 - inverted_speed
                speeds_of_transitions.append(round(original_speed, 2))  # Ensure proper rounding
        
        # Reconstructing distances_to_next_frames with exact values
        distances_to_next_frames = []
        for i in range(1, len(dynamic_frame_distribution_values)):
            distances_to_next_frames.append(dynamic_frame_distribution_values[i] - dynamic_frame_distribution_values[i-1])
        
        return strength_of_frames,freedoms_between_frames, speeds_of_transitions

    def identify_type_of_motion_context(context_length, context_stride, context_overlap):
        # Given the context settings, identify the type of motion context
        if context_stride == 1 and context_overlap == 2:
            return "Low"
        elif context_stride == 2 and context_overlap == 4:
            return "Standard"
        elif context_stride == 4 and context_overlap == 4:
            return "High"
        else:
            return "Unknown"  # Fallback case if the inputs do not match expected values
        
    def calculate_strength_of_adherence(multipled_base_end_percent):
        return multipled_base_end_percent / (0.05 * 10)

    def reverse_frame_prompts_formatting(formatted_prompts):
        # Extract frame number and prompt pairs using a regular expression
        prompt_pairs = re.findall(r'\"(\d+\.\d+)\":\s*\"(.*?)\"', formatted_prompts)
        
        # Initialize an empty list to collect prompts
        original_prompts = [prompt for frame, prompt in prompt_pairs]
        
        return original_prompts


    def reverse_motion_strengths_formatting(formatted_motions, buffer):
        # Extract frame number and motion strength pairs using a regular expression
        motion_pairs = re.findall(r'(\d+):\((.*?)\)', formatted_motions)
        
        # Convert extracted pairs back to the original format, adjusting frame numbers
        original_motions = []
        for frame, strength in motion_pairs:
            original_frame = int(frame) - buffer  # Subtract buffer to get original frame number
            original_strength = float(strength)  # Convert strength back to float
            # Ensure the motion is appended in the correct order based on original frame numbers
            original_motions.append(original_strength)
        
        return original_motions
    

    def safe_eval(input_data):
        if isinstance(input_data, str):
            try:
                return ast.literal_eval(input_data)
            except ValueError:
                # Handle the case where the string cannot be parsed
                return input_data
        else:
            return input_data

    dynamic_strength_values = safe_eval(dynamic_strength_values)
    dynamic_key_frame_influence_values = safe_eval(dynamic_key_frame_influence_values)
    dynamic_frame_distribution_values = safe_eval(dynamic_frame_distribution_values)

    context_length = int(context_length)
    context_stride = int(context_stride)
    context_overlap = int(context_overlap)
    multipled_base_end_percent = float(multipled_base_end_percent)    

    # Step 1: Reverse dynamic_strength_values and dynamic_key_frame_influence_values

    strength_of_frames, freedoms_between_frames, speeds_of_transitions  = reverse_transform(dynamic_strength_values, dynamic_key_frame_influence_values, dynamic_frame_distribution_values)
    
    # Step 2: Reverse dynamic_frame_distribution_values to distances_to_next_frames
    distances_to_next_frames = [round((dynamic_frame_distribution_values[i] - dynamic_frame_distribution_values[i-1]) / 16, 2) for i in range(1, len(dynamic_frame_distribution_values))]
    
    # Step 3: Identify type_of_motion_context
    type_of_motion_context = identify_type_of_motion_context(context_length, context_stride, context_overlap)
    
    # Step 4: Calculate strength_of_adherence from multipled_base_end_percent
    strength_of_adherence = calculate_strength_of_adherence(multipled_base_end_percent)
    
    # Step 5: Reverse frame prompts formatting

    individual_prompts = reverse_frame_prompts_formatting(formatted_individual_prompts) 

    individual_negative_prompts = reverse_frame_prompts_formatting(formatted_individual_negative_prompts)

    # Step 6: Reverse motion strengths formatting
    motions_during_frames = reverse_motion_strengths_formatting(formatted_motions, buffer)

    return strength_of_frames, freedoms_between_frames, speeds_of_transitions, distances_to_next_frames, type_of_motion_context, strength_of_adherence, individual_prompts, individual_negative_prompts, motions_during_frames
    
