from typing import List
import time
import ast
import streamlit as st
from shared.constants import AnimationStyleType, AnimationToolType, STEERABLE_MOTION_WORKFLOWS
import time
from ui_components.constants import DEFAULT_SHOT_MOTION_VALUES
from ui_components.methods.ml_methods import generate_sm_video
from ui_components.widgets.sm_animation_style_element import (
    animation_sidebar,
    individual_frame_settings_element,
    select_motion_lora_element,
    select_sd_model_element,
    video_motion_settings,
)
from ui_components.models import InternalFileObject, InternalShotObject
from ui_components.methods.animation_style_methods import (
    is_inference_enabled,
    toggle_generate_inference,
    transform_data,
    update_session_state_with_animation_details,
    update_session_state_with_dc_details,
)
from utils import st_memory
from utils.state_refresh import refresh_app
from utils.data_repo.data_repo import DataRepo

DEFAULT_SM_MODEL = "dreamshaper_8.safetensors"


def sm_video_rendering_page(shot_uuid, img_list: List[InternalFileObject], column1, column2):
    data_repo = DataRepo()
    shot: InternalShotObject = data_repo.get_shot_from_uuid(shot_uuid)

    settings = {
        "animation_tool": AnimationToolType.ANIMATEDIFF.value,
    }
    shot_meta_data = {}

    with st.container():

        # ----------- HEADER OPTIONS -------------
        header_col_1, _, header_col_3, header_col_4 = st.columns(
            [1.0, 1.5, 1.0, 1.0]
        )  # btns defined at the bottom of the UI

        # ----------- INDIVIDUAL FRAME SETTINGS -----------
        (
            strength_of_frames,
            distances_to_next_frames,
            speeds_of_transitions,
            freedoms_between_frames,
            individual_prompts,
            individual_negative_prompts,
            motions_during_frames,
        ) = individual_frame_settings_element(shot_uuid, img_list)

        # ----------- SELECT SD MODEL -----------
        sd_model, model_files = select_sd_model_element(shot_uuid, DEFAULT_SM_MODEL)

        # ----------- SELECT MOTION LORA ------------
        lora_data = select_motion_lora_element(shot_uuid, model_files)

        # ----------- OTHER SETTINGS ------------
        (
            strength_of_adherence,
            overall_positive_prompt,
            overall_negative_prompt,
            type_of_motion_context,
            allow_for_looping,
            high_detail_mode,
            stabilise_motion,
        ) = video_motion_settings(shot_uuid, img_list)

        type_of_frame_distribution = "dynamic"
        type_of_key_frame_influence = "dynamic"
        type_of_strength_distribution = "dynamic"
        linear_frame_distribution_value = 16
        linear_key_frame_influence_value = 1.0
        linear_cn_strength_value = 1.0
        relative_ipadapter_strength = 1.0
        relative_cn_strength = 0.0
        project_settings = data_repo.get_project_setting(shot.project.uuid)
        width = project_settings.width
        height = project_settings.height
        img_dimension = f"{width}x{height}"
        motion_scale = 1.3
        interpolation_style = "ease-in-out"
        buffer = 4

        (
            dynamic_strength_values,
            dynamic_key_frame_influence_values,
            dynamic_frame_distribution_values,
            context_length,
            context_stride,
            context_overlap,
            multipled_base_end_percent,
            multipled_base_adapter_strength,
            prompt_travel,
            negative_prompt_travel,
            motion_scales,
        ) = transform_data(
            strength_of_frames,
            freedoms_between_frames,
            speeds_of_transitions,
            distances_to_next_frames,
            type_of_motion_context,
            strength_of_adherence,
            individual_prompts,
            individual_negative_prompts,
            buffer,
            motions_during_frames,
        )

        settings.update(
            ckpt=sd_model,
            width=width,
            height=height,
            buffer=4,
            motion_scale=motion_scale,
            motion_scales=motion_scales,
            high_detail_mode=high_detail_mode,
            image_dimension=img_dimension,
            output_format="video/h264-mp4",
            prompt=overall_positive_prompt,
            allow_for_looping=allow_for_looping,
            negative_prompt=overall_negative_prompt,
            interpolation_type=interpolation_style,
            stmfnet_multiplier=2,
            relative_ipadapter_strength=relative_ipadapter_strength,
            relative_cn_strength=relative_cn_strength,
            type_of_strength_distribution=type_of_strength_distribution,
            linear_strength_value=str(linear_cn_strength_value),
            dynamic_strength_values=str(dynamic_strength_values),
            linear_frame_distribution_value=linear_frame_distribution_value,
            dynamic_frame_distribution_values=dynamic_frame_distribution_values,
            type_of_frame_distribution=type_of_frame_distribution,
            type_of_key_frame_influence=type_of_key_frame_influence,
            linear_key_frame_influence_value=float(linear_key_frame_influence_value),
            dynamic_key_frame_influence_values=dynamic_key_frame_influence_values,
            normalise_speed=True,
            ipadapter_noise=0.3,
            animation_style=AnimationStyleType.CREATIVE_INTERPOLATION.value,
            context_length=context_length,
            context_stride=context_stride,
            context_overlap=context_overlap,
            multipled_base_end_percent=multipled_base_end_percent,
            multipled_base_adapter_strength=multipled_base_adapter_strength,
            individual_prompts=prompt_travel,
            individual_negative_prompts=negative_prompt_travel,
            animation_stype=AnimationStyleType.CREATIVE_INTERPOLATION.value,
            max_frames=str(dynamic_frame_distribution_values[-1]),
            stabilise_motion=stabilise_motion,
            lora_data=lora_data,
            shot_data=shot_meta_data,
            pil_img_structure_control_image=st.session_state[
                f"structure_control_image_{shot.uuid}"
            ],  # this is a PIL object
            strength_of_structure_control_image=st.session_state[
                f"strength_of_structure_control_image_{shot.uuid}"
            ],
            filename_prefix="AD_",
        )

        st.markdown("***")
        st.markdown("##### Generation Settings")

        filtered_and_sorted_workflows = sorted(
            (workflow for workflow in STEERABLE_MOTION_WORKFLOWS if workflow["display"]),
            key=lambda x: x["order"],
        )

        generation_types = [workflow["name"] for workflow in filtered_and_sorted_workflows]

        footer1, footer2 = st.columns([1.5, 1])
        with footer1:
            number_of_generation_steps = st_memory.number_input(
                "Number of generation steps:",                
                key=f"number_of_generation_steps_{shot.uuid}",
                min_value=5,
                max_value=30,
                step=1,
            )
            type_of_generation = st.radio(
                "Workflow variant:",
                options=generation_types,
                key="creative_interpolation_type",
                horizontal=True,
                index=st.session_state.get(f"type_of_generation_index_{shot.uuid}", 0),
                help="""
                
                    **Slurshy Realistiche**: good for simple realistic motion.

                    **Smooth n' Steady**: good for slow, smooth transitions. 
                    
                    **Chocky Realistiche**: good for realistic motion and chaotic transitions. 

                    **Liquidy Loop**: good for liquid-like motion with slick transitions. Also loops!
                    
                    **Fast With A Price**: runs fast but with a lot of detail loss.
                    
                    **Rad Attack**: good for realistic motion but with a lot of detail loss.
                    
                    """,
            )

            if (
                type_of_generation
                != generation_types[st.session_state.get(f"type_of_generation_index_{shot.uuid}", 0)]
            ):
                st.session_state[f"type_of_generation_index_{shot.uuid}"] = generation_types.index(
                    type_of_generation
                )
                refresh_app()

        with footer2:
            st.info(
                f"Each has a unique type of motion and adherence. You can an example of each of them in action [here](https://youtu.be/zu1IbdavW_4)."
            )

        generate_vid_inf_tag = "generate_vid"
        manual_save_inf_tag = "manual_save"

        st.write("")
        animate_col_1, _, _ = st.columns([2, 1, 1])
        with animate_col_1:
            variant_count = 1

            if is_inference_enabled(generate_vid_inf_tag) or is_inference_enabled(manual_save_inf_tag):

                st.session_state['auto_refresh'] = False
                # last keyframe position * 16
                duration = float(dynamic_frame_distribution_values[-1] / 16)
                data_repo.update_shot(uuid=shot_uuid, duration=duration)

                # converting PIL imgs to InternalFileObject
                from ui_components.methods.common_methods import save_new_image

                key = "pil_img_structure_control_image"
                image = None
                if settings[key]:
                    image = save_new_image(settings[key], shot.project.uuid)
                    del settings[key]
                    new_key = key.replace("pil_img_", "") + "_uuid"
                    settings[new_key] = image.uuid

                # print("******************* ", st.session_state.get(f"{shot_uuid}_preview_mode", False))
                if st.session_state.get(f"{shot_uuid}_preview_mode", False):
                    start_frame, end_frame = st.session_state.get(f"frames_to_preview_{shot_uuid}", (1, 3))
                    preview_length = end_frame - start_frame + 1

                    
                    img_list = img_list[start_frame-1:end_frame]
                    # Calculate the offset based on the first number in dynamic_frame_distribution_values
                    frame_offset = settings["dynamic_frame_distribution_values"][start_frame-1]

                    # Adjust dynamic_strength_values
                    _t = ast.literal_eval(settings["dynamic_strength_values"])[start_frame-1:end_frame]
                    settings["dynamic_strength_values"] = f"[{', '.join(repr(t) for t in _t)}]"
                    
                    # Adjust dynamic_frame_distribution_values
                    settings["dynamic_frame_distribution_values"] = [
                        v - frame_offset for v in settings["dynamic_frame_distribution_values"][start_frame-1:end_frame]
                    ]
                    
                    # Adjust dynamic_key_frame_influence_values
                    settings["dynamic_key_frame_influence_values"] = settings[
                        "dynamic_key_frame_influence_values"
                    ][start_frame-1:end_frame]
                    
                    # Adjust individual prompts and negative prompts
                    individual_prompts_dict = ast.literal_eval('{' + settings["individual_prompts"] + '}')
                    individual_negative_prompts_dict = ast.literal_eval('{' + settings["individual_negative_prompts"] + '}')
                    
                    new_individual_prompts = {str(int(k) - frame_offset): v for k, v in individual_prompts_dict.items() if frame_offset <= int(k) < frame_offset + preview_length * 26}
                    new_individual_negative_prompts = {str(int(k) - frame_offset): v for k, v in individual_negative_prompts_dict.items() if frame_offset <= int(k) < frame_offset + preview_length * 26}
                    
                    settings["individual_prompts"] = ', '.join(f'"{k}": "{v}"' for k, v in new_individual_prompts.items())
                    settings["individual_negative_prompts"] = ', '.join(f'"{k}": "{v}"' for k, v in new_individual_negative_prompts.items())

                    # Adjust other settings
                    settings["strength_of_frames"] = strength_of_frames[start_frame-1:end_frame]
                    settings["speeds_of_transitions"] = speeds_of_transitions[start_frame-1:end_frame]
                    settings["distances_to_next_frames"] = distances_to_next_frames[start_frame-1:end_frame]
                    settings["freedoms_between_frames"] = freedoms_between_frames[start_frame-1:end_frame]
                    settings["motions_during_frames"] = motions_during_frames[start_frame-1:end_frame]

                    # Update the local variables to match the settings
                    
                    strength_of_frames = settings["strength_of_frames"]
                    speeds_of_transitions = settings["speeds_of_transitions"]
                    distances_to_next_frames = settings["distances_to_next_frames"]
                    freedoms_between_frames = settings["freedoms_between_frames"]
                    motions_during_frames = settings["motions_during_frames"]
                    
                    individual_prompts = [v for v in new_individual_prompts.values()]
                    individual_negative_prompts = [v for v in new_individual_negative_prompts.values()]
                    
                    settings["inference_type"] = "preview"
                    trigger_shot_update = False

                else:
                    trigger_shot_update = True

                shot_data = update_session_state_with_animation_details(
                        shot_uuid,
                        img_list,
                        strength_of_frames,
                        distances_to_next_frames,
                        speeds_of_transitions,
                        freedoms_between_frames,
                        motions_during_frames,
                        individual_prompts,
                        individual_negative_prompts,
                        lora_data,
                        DEFAULT_SM_MODEL,
                        high_detail_mode,
                        image.uuid if image else None,
                        settings["strength_of_structure_control_image"],
                        next(
                            (
                                index
                                for index, workflow in enumerate(filtered_and_sorted_workflows)
                                if workflow["name"] == type_of_generation
                            ),
                            0,
                        ),
                        stabilise_motion=stabilise_motion,
                        trigger_shot_update=trigger_shot_update,
                    )
                
                settings.update(shot_data=shot_data)
                settings.update(number_of_generation_steps=number_of_generation_steps)
                settings.update(type_of_generation=type_of_generation)
                settings.update(filename_prefix="AD_")

                st.success(
                    "Generating clip - see status in the Generation Log in the sidebar. Press 'Refresh log' to update."
                )

                positive_prompt = ""
                append_to_prompt = ""
                for idx, img in enumerate(img_list):
                    if img.location:
                        b = img.inference_params
                        prompt = b.get("prompt", "") if b else ""
                        prompt += append_to_prompt
                        frame_prompt = f"{idx * linear_frame_distribution_value}_" + prompt
                        positive_prompt += ":" + frame_prompt if positive_prompt else frame_prompt
                    else:
                        st.error("Please generate primary images")
                        time.sleep(0.7)
                        refresh_app()

                if f"{shot_uuid}_backlog_enabled" not in st.session_state:
                    st.session_state[f"{shot_uuid}_backlog_enabled"] = False

                if is_inference_enabled(generate_vid_inf_tag):
                    generate_sm_video(
                        shot_uuid,
                        settings,
                        variant_count,
                        st.session_state[f"{shot_uuid}_backlog_enabled"],
                        img_list,
                    )

                updated_additional_params = {
                    f"{shot_uuid}_backlog_enabled": False,
                    f"{shot_uuid}_preview_mode": False,
                }

                position = (
                    generate_vid_inf_tag
                    if is_inference_enabled(generate_vid_inf_tag)
                    else manual_save_inf_tag
                )
                toggle_generate_inference(position, **updated_additional_params)
                st.session_state['auto_refresh'] = True
                refresh_app()

            preview_mode = st_memory.checkbox(
                label="Preview mode",
                key=f"{shot_uuid}_gen_preview_mode",
                help="Generates a preview video only using the first 3 images",
                value=False,
            )

            if preview_mode:
                # take a range of frames from the user
                frames_to_preview = st_memory.slider(
                    "Frames to preview:",
                    min_value=1,
                    max_value=len(img_list),
                    value=(1, min(3, len(img_list))),
                    key=f"frames_to_preview_{shot_uuid}"
                )
                start_frame, end_frame = st.session_state[f"frames_to_preview_{shot_uuid}"]
                preview_frames = img_list[start_frame-1:end_frame]
                
                num_columns = min(3, len(preview_frames))
                preview_columns = st.columns(num_columns)
                
                for i, frame in enumerate(preview_frames):
                    with preview_columns[i % num_columns]:
                        st.image(frame.location, use_column_width=True)

                if len(preview_frames) == 1:
                    st.error("You need at least 2 frames to preview")
                

    
            btn1, btn2, _ = st.columns([1, 1, 1])
            additional_params = {
                f"{shot_uuid}_backlog_enabled": False,
                f"{shot_uuid}_preview_mode": preview_mode,
            }

            with btn1:
                help = ""
                st.button(
                    "Add to queue",
                    key="generate_animation_clip",
                    disabled=False,
                    help=help,
                    on_click=lambda: toggle_generate_inference(generate_vid_inf_tag, **additional_params),
                    type="primary",
                    use_container_width=True,
                )

            backlog_update = {f"{shot_uuid}_backlog_enabled": True}
            with btn2:
                backlog_help = "This will add the new video generation in the backlog"
                st.button(
                    "Add to backlog",
                    key="generate_animation_clip_backlog",
                    disabled=False,
                    help=backlog_help,
                    on_click=lambda: toggle_generate_inference(generate_vid_inf_tag, **backlog_update),
                    type="secondary",
                )

            with column1:
                if st.button("Reset to default", use_container_width=True, key="reset_to_default"):
                    for idx, _ in enumerate(img_list):
                        for k, v in DEFAULT_SHOT_MOTION_VALUES.items():
                            st.session_state[f"{k}_{shot_uuid}_{idx}"] = v

                    st.success("All frames have been reset to default values.")
                    refresh_app()
                st.write("")

            with column2:
                if st.button(
                    "Save current settings",
                    key="save_current_settings",
                    use_container_width=True,
                    help="Settings will also be saved when you generate the animation.",
                ):
                    st.success("Settings saved successfully")
                    toggle_generate_inference(manual_save_inf_tag, **additional_params)
                    refresh_app()

        # --------------- SIDEBAR ---------------------
        animation_sidebar(
            shot_uuid,
            img_list,
            type_of_frame_distribution,
            dynamic_frame_distribution_values,
            linear_frame_distribution_value,
            type_of_strength_distribution,
            dynamic_strength_values,
            linear_cn_strength_value,
            type_of_key_frame_influence,
            dynamic_key_frame_influence_values,
            linear_key_frame_influence_value,
            strength_of_frames,
            distances_to_next_frames,
            speeds_of_transitions,
            freedoms_between_frames,
            motions_during_frames,
            individual_prompts,
            individual_negative_prompts,
            DEFAULT_SM_MODEL,
        )


def two_img_realistic_interpolation_page(shot_uuid, img_list: List[InternalFileObject]):
    if not (img_list and len(img_list) >= 2):
        st.error("You need two images for this interpolation")
        return

    data_repo = DataRepo()
    shot = data_repo.get_shot_from_uuid(shot_uuid)

    settings = {}
    st.markdown("***")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(img_list[0].location, use_column_width=True)

    with col3:
        st.image(img_list[1].location, use_column_width=True)

    with col2:
        if f"video_desc_{shot_uuid}" not in st.session_state:
            st.session_state[f"video_desc_{shot_uuid}"] = ""
        description_of_motion = st.text_area(
            "Describe the motion you want between the frames:",
            key=f"description_of_motion_{shot.uuid}",
            value=st.session_state[f"video_desc_{shot_uuid}"],
        )
        st.info("This is very important and will likely require some iteration.")
        st.info("NOTE: The model for this animation is 10.5 GB in size, which can take some time to download")

    variant_count = 1  # Assuming a default value for variant_count, adjust as necessary
    position = "dynamiccrafter"

    if (
        f"{position}_generate_inference" in st.session_state
        and st.session_state[f"{position}_generate_inference"]
    ):
        st.success(
            "Generating clip - see status in the Generation Log in the sidebar. Press 'Refresh log' to update."
        )
        # Assuming the logic to generate the clip based on two images, the described motion, and fixed duration
        duration = 4  # Fixed duration of 4 seconds
        data_repo.update_shot(uuid=shot.uuid, duration=duration)

        project_settings = data_repo.get_project_setting(shot.project.uuid)
        meta_data = update_session_state_with_dc_details(
            shot_uuid,
            img_list,
            description_of_motion,
        )
        settings.update(shot_data=meta_data)
        settings.update(
            duration=duration,
            animation_style=AnimationStyleType.DIRECT_MORPHING.value,
            output_format="video/h264-mp4",
            width=project_settings.width,
            height=project_settings.height,
            prompt=description_of_motion,
        )

        generate_sm_video(
            shot_uuid,
            settings,
            variant_count,
            st.session_state[f"{shot_uuid}_backlog_enabled"],
            img_list,
        )

        backlog_update = {f"{shot_uuid}_backlog_enabled": False}
        toggle_generate_inference(position, **backlog_update)
        refresh_app()

    # Buttons for adding to queue or backlog, assuming these are still relevant
    st.markdown("***")
    btn1, btn2, btn3 = st.columns([1, 1, 1])
    backlog_no_update = {f"{shot_uuid}_backlog_enabled": False}

    with btn1:
        st.button(
            "Add to queue",
            key="generate_animation_clip",
            disabled=False,
            help="Generate the interpolation clip based on the two images and described motion.",
            on_click=lambda: toggle_generate_inference(position, **backlog_no_update),
            type="primary",
            use_container_width=True,
        )

    backlog_update = {f"{shot_uuid}_backlog_enabled": True}
    with btn2:
        st.button(
            "Add to backlog",
            key="generate_animation_clip_backlog",
            disabled=False,
            help="Add the 2-Image Realistic Interpolation to the backlog.",
            on_click=lambda: toggle_generate_inference(position, **backlog_update),
            type="secondary",
        )
