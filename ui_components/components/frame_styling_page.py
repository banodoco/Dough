import streamlit as st
from streamlit_image_comparison import image_comparison
import time
import pandas as pd
from repository.local_repo.csv_repo import get_app_settings, get_project_settings
from ui_components.common_methods import create_gif_preview, delete_frame, get_model_details_from_csv, get_timing_details, promote_image_variant, trigger_restyling_process

def frame_styling_page(mainheader2, project_name):
    timing_details = get_timing_details(project_name)
    with mainheader2:
        with st.expander("💡 How frame styling works"):
            st.info("On the left, there are a bunch of differnet models and processes you can use to style frames. You can even use combinatinos of models through custom pipelines or by running them one after another. We recommend experimenting on 1-2 frames before doing bulk runs for the sake of efficiency.")

    if "project_settings" not in st.session_state:
        st.session_state['project_settings'] = get_project_settings(project_name)
        print("HERE LAD")
    print(st.session_state['project_settings'])
                            
    if "strength" not in st.session_state:                    
        st.session_state['strength'] = st.session_state['project_settings']["last_strength"]
        st.session_state['prompt_value'] = st.session_state['project_settings']["last_prompt"]
        st.session_state['model'] = st.session_state['project_settings']["last_model"]
        st.session_state['custom_pipeline'] = st.session_state['project_settings']["last_custom_pipeline"]
        st.session_state['negative_prompt_value'] = st.session_state['project_settings']["last_negative_prompt"]
        st.session_state['guidance_scale'] = st.session_state['project_settings']["last_guidance_scale"]
        st.session_state['seed'] = st.session_state['project_settings']["last_seed"]
        st.session_state['num_inference_steps'] = st.session_state['project_settings']["last_num_inference_steps"]
        st.session_state['which_stage_to_run_on'] = st.session_state['project_settings']["last_which_stage_to_run_on"]
        st.session_state['show_comparison'] = "Don't show"
                            

    if "which_image" not in st.session_state:
        st.session_state['which_image'] = 0
                                

    if 'frame_styling_view_type' not in st.session_state:
        st.session_state['frame_styling_view_type'] = "List View"
        st.session_state['frame_styling_view_type_index'] = 0


    if timing_details == []:
        st.info("You need to select and load key frames first in the Key Frame Selection section.")                            
    else:
        top1, top2, top3 = st.columns([3,1,2])
        with top1:
            view_types = ["List View","Single Frame"]
            st.session_state['frame_styling_view_type'] = st.radio("View type:", view_types, key="which_view_type", horizontal=True, index=st.session_state['frame_styling_view_type_index'])                        
            if view_types.index(st.session_state['frame_styling_view_type']) != st.session_state['frame_styling_view_type_index']:
                st.session_state['frame_styling_view_type_index'] = view_types.index(st.session_state['frame_styling_view_type'])
                st.experimental_rerun()
                                                            
        with top2:
            st.write("")


        if st.session_state['frame_styling_view_type'] == "Single Frame":
            with top3:
                st.session_state['show_comparison'] = st.radio("Show comparison to original", options=["Don't show", "Show"], horizontal=True)
                
                                    
            f1, f2, f3  = st.columns([1,4,1])
            
            with f1:
                st.session_state['which_image'] = st.number_input(f"Key frame # (out of {len(timing_details)-1})", 0, len(timing_details)-1, value=st.session_state['which_image_value'], step=1, key="which_image_selector")
                if st.session_state['which_image_value'] != st.session_state['which_image']:
                    st.session_state['which_image_value'] = st.session_state['which_image']
                    st.experimental_rerun()
            if timing_details[st.session_state['which_image']]["alternative_images"] != "":                                                                                       
                variants = timing_details[st.session_state['which_image']]["alternative_images"]
                number_of_variants = len(variants)
                current_variant = int(timing_details[st.session_state['which_image']]["primary_image"])                                                                                                
                which_variant = current_variant     
                with f2:
                    which_variant = st.radio(f'Main variant = {current_variant}', range(number_of_variants), index=current_variant, horizontal = True, key = f"Main variant for {st.session_state['which_image']}")
                with f3:
                    if which_variant == current_variant:   
                        st.write("")                                   
                        st.success("Main variant")
                    else:
                        st.write("")
                        if st.button(f"Promote Variant #{which_variant}", key=f"Promote Variant #{which_variant} for {st.session_state['which_image']}", help="Promote this variant to the primary image"):
                            promote_image_variant(st.session_state['which_image'], project_name, which_variant)
                            time.sleep(0.5)
                            st.experimental_rerun()
            
            
            if st.session_state['show_comparison'] == "Don't show":
                if timing_details[st.session_state['which_image']]["alternative_images"] != "":
                    st.image(variants[which_variant], use_column_width=True)   
                else:
                    st.image('https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png', use_column_width=True)   
            else:
                if timing_details[st.session_state['which_image']]["alternative_images"] != "":
                    img2=variants[which_variant]
                else:
                    img2='https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png'          
                image_comparison(starting_position=50,
                    img1=timing_details[st.session_state['which_image']]["source_image"],
                    img2=img2,make_responsive=False)
                
            

        if 'index_of_last_model' not in st.session_state:
            st.session_state['index_of_last_model'] = 0



        if len(timing_details) == 0:
            st.info("You first need to select key frames at the Key Frame Selection stage.")

        st.sidebar.header("Restyle Frames")   
        if 'index_of_which_stage_to_run_on' not in st.session_state:                        
            st.session_state['index_of_which_stage_to_run_on'] = 0
        stages = ["Extracted Key Frames", "Current Main Variants"]
        st.session_state['which_stage_to_run_on'] = st.sidebar.radio("What stage of images would you like to run styling on?", options=stages, horizontal=True, index =st.session_state['index_of_which_stage_to_run_on'] , help="Extracted frames means the original frames from the video.")                                                                                     
        if stages.index(st.session_state['which_stage_to_run_on']) != st.session_state['index_of_which_stage_to_run_on']:
            st.session_state['index_of_which_stage_to_run_on'] = stages.index(st.session_state['which_stage_to_run_on'])
            st.experimental_rerun()

        custom_pipelines = ["None","Mystique"]                   
        if 'index_of_last_custom_pipeline' not in st.session_state:
            st.session_state['index_of_last_custom_pipeline'] = 0        
        st.session_state['custom_pipeline'] = st.sidebar.selectbox(f"Custom Pipeline:", custom_pipelines, index=st.session_state['index_of_last_custom_pipeline'])
        if custom_pipelines.index(st.session_state['custom_pipeline']) != st.session_state['index_of_last_custom_pipeline']:
            st.session_state['index_of_last_custom_pipeline'] = custom_pipelines.index(st.session_state['custom_pipeline'])
            st.experimental_rerun()

        if st.session_state['custom_pipeline'] == "Mystique":
            if st.session_state['index_of_last_model'] > 1:
                st.session_state['index_of_last_model'] = 0       
                st.experimental_rerun()           
            with st.sidebar.expander("Mystique is a custom pipeline that uses a multiple models to generate a consistent character and style transformation."):
                st.markdown("## How to use the Mystique pipeline")                
                st.markdown("1. Create a fine-tined model in the Custom Model section of the app - we recommend Dreambooth for character transformations.")
                st.markdown("2. It's best to include a detailed prompt. We recommend taking an example input image and running it through the Prompt Finder")
                st.markdown("3. Use [expression], [location], [mouth], and [looking] tags to vary the expression and location of the character dynamically if that changes throughout the clip. Varying this in the prompt will make the character look more natural - especially useful if the character is speaking.")
                st.markdown("4. In our experience, the best strength for coherent character transformations is 0.25-0.3 - any more than this and details like eye position change.")  
            models = ["LoRA","Dreambooth"]                                     
            st.session_state['model'] = st.sidebar.selectbox(f"Which type of model is trained on your character?", models, index=st.session_state['index_of_last_model'])                    
            if st.session_state['index_of_last_model'] != models.index(st.session_state['model']):
                st.session_state['index_of_last_model'] = models.index(st.session_state['model'])
                st.experimental_rerun()                          
        else:
            models = ['stable-diffusion-img2img-v2.1', 'depth2img', 'pix2pix', 'controlnet', 'Dreambooth', 'LoRA','StyleGAN-NADA']            
            st.session_state['model'] = st.sidebar.selectbox(f"Which model would you like to use?", models, index=st.session_state['index_of_last_model'])                    
            if st.session_state['index_of_last_model'] != models.index(st.session_state['model']):
                st.session_state['index_of_last_model'] = models.index(st.session_state['model'])
                st.experimental_rerun() 
                
        
        if st.session_state['model'] == "controlnet":   
            controlnet_adapter_types = ["normal", "canny", "hed", "scribble", "seg", "hough", "depth2img", "pose"]
            if 'index_of_controlnet_adapter_type' not in st.session_state:
                st.session_state['index_of_controlnet_adapter_type'] = 0
            st.session_state['adapter_type'] = st.sidebar.selectbox(f"Adapter Type",controlnet_adapter_types, index=st.session_state['index_of_controlnet_adapter_type'])
            if st.session_state['index_of_controlnet_adapter_type'] != controlnet_adapter_types.index(st.session_state['adapter_type']):
                st.session_state['index_of_controlnet_adapter_type'] = controlnet_adapter_types.index(st.session_state['adapter_type'])
                st.experimental_rerun()
            custom_models = []           
        elif st.session_state['model'] == "LoRA": 
            if 'index_of_lora_model_1' not in st.session_state:
                st.session_state['index_of_lora_model_1'] = 0
                st.session_state['index_of_lora_model_2'] = 0
                st.session_state['index_of_lora_model_3'] = 0
            df = pd.read_csv('models.csv')
            filtered_df = df[df.iloc[:, 5] == 'LoRA']
            lora_model_list = filtered_df.iloc[:, 0].tolist()
            lora_model_list.insert(0, '')
            st.session_state['lora_model_1'] = st.sidebar.selectbox(f"LoRA Model 1", lora_model_list, index=st.session_state['index_of_lora_model_1'])
            if st.session_state['index_of_lora_model_1'] != lora_model_list.index(st.session_state['lora_model_1']):
                st.session_state['index_of_lora_model_1'] = lora_model_list.index(st.session_state['lora_model_1'])
                st.experimental_rerun()
            st.session_state['lora_model_2'] = st.sidebar.selectbox(f"LoRA Model 2", lora_model_list, index=st.session_state['index_of_lora_model_2'])
            if st.session_state['index_of_lora_model_2'] != lora_model_list.index(st.session_state['lora_model_2']):
                st.session_state['index_of_lora_model_2'] = lora_model_list.index(st.session_state['lora_model_2'])
                st.experimental_rerun()
            st.session_state['lora_model_3'] = st.sidebar.selectbox(f"LoRA Model 3", lora_model_list, index=st.session_state['index_of_lora_model_3'])
            if st.session_state['index_of_lora_model_3'] != lora_model_list.index(st.session_state['lora_model_3']):
                st.session_state['index_of_lora_model_3'] = lora_model_list.index(st.session_state['lora_model_3'])                     
                st.experimental_rerun()
            custom_models = [st.session_state['lora_model_1'], st.session_state['lora_model_2'], st.session_state['lora_model_3']]                    
            st.sidebar.info("You can reference each model in your prompt using the following keywords: <1>, <2>, <3> - for example '<1> in the style of <2>.")
            lora_adapter_types = ['sketch', 'seg', 'keypose', 'depth', None]
            if "index_of_lora_adapter_type" not in st.session_state:
                st.session_state['index_of_lora_adapter_type'] = 0
            st.session_state['adapter_type'] = st.sidebar.selectbox(f"Adapter Type:", lora_adapter_types, help="This is the method through the model will infer the shape of the object. ", index=st.session_state['index_of_lora_adapter_type'])
            if st.session_state['index_of_lora_adapter_type'] != lora_adapter_types.index(st.session_state['adapter_type']):
                st.session_state['index_of_lora_adapter_type'] = lora_adapter_types.index(st.session_state['adapter_type'])
        elif st.session_state['model'] == "Dreambooth":
            df = pd.read_csv('models.csv')
            filtered_df = df[df.iloc[:, 5] == 'Dreambooth']
            dreambooth_model_list = filtered_df.iloc[:, 0].tolist()
            if 'index_of_dreambooth_model' not in st.session_state:
                st.session_state['index_of_dreambooth_model'] = 0
            custom_models = st.sidebar.selectbox(f"Dreambooth Model", dreambooth_model_list, index=st.session_state['index_of_dreambooth_model'])
            if st.session_state['index_of_dreambooth_model'] != dreambooth_model_list.index(custom_models):
                st.session_state['index_of_dreambooth_model'] = dreambooth_model_list.index(custom_models)                                    
        else:
            custom_models = []
            st.session_state['adapter_type'] = "N"
        
        if st.session_state['model'] == "StyleGAN-NADA":
            st.sidebar.warning("StyleGAN-NADA is a custom model that uses StyleGAN to generate a consistent character and style transformation. It only works for square images.")
            st.session_state['prompt'] = st.sidebar.selectbox("What style would you like to apply to the character?", ['base', 'mona_lisa', 'modigliani', 'cubism', 'elf', 'sketch_hq', 'thomas', 'thanos', 'simpson', 'witcher', 'edvard_munch', 'ukiyoe', 'botero', 'shrek', 'joker', 'pixar', 'zombie', 'werewolf', 'groot', 'ssj', 'rick_morty_cartoon', 'anime', 'white_walker', 'zuckerberg', 'disney_princess', 'all', 'list'])
            st.session_state['strength'] = 0.5
            st.session_state['guidance_scale'] = 7.5
            st.session_state['seed'] = int(0)
            st.session_state['num_inference_steps'] = int(50)
                        
        else:
            st.session_state['prompt'] = st.sidebar.text_area(f"Prompt", label_visibility="visible", value=st.session_state['prompt_value'],height=150)
            if st.session_state['prompt'] != st.session_state['prompt_value']:
                st.session_state['prompt_value'] = st.session_state['prompt']
                st.experimental_rerun()
            with st.sidebar.expander("💡 Learn about dynamic prompting"):
                st.markdown("## Why and how to use dynamic prompting")
                st.markdown("Why:")
                st.markdown("Dynamic prompting allows you to automatically vary the prompt throughout the clip based on changing features in the source image. This makes the output match the input more closely and makes character transformations look more natural.")
                st.markdown("How:")
                st.markdown("You can include the following tags in the prompt to vary the prompt dynamically: [expression], [location], [mouth], and [looking]")
            if st.session_state['model'] == "Dreambooth":
                model_details = get_model_details_from_csv(custom_models)
                st.sidebar.info(f"Must include '{model_details['keyword']}' to run this model")   
                if model_details['controller_type'] != "":                    
                    st.session_state['adapter_type']  = st.sidebar.selectbox(f"Would you like to use the {model_details['controller_type']} controller?", ['Yes', 'No'])
                else:
                    st.session_state['adapter_type']  = "No"

            else:
                if st.session_state['model'] == "pix2pix":
                    st.sidebar.info("In our experience, setting the seed to 87870, and the guidance scale to 7.5 gets consistently good results. You can set this in advanced settings.")                    
            st.session_state['strength'] = st.sidebar.number_input(f"Strength", value=float(st.session_state['strength']), min_value=0.0, max_value=1.0, step=0.01)
            
            with st.sidebar.expander("Advanced settings 😏"):
                st.session_state['negative_prompt'] = st.text_area(f"Negative prompt", value=st.session_state['negative_prompt_value'], label_visibility="visible")
                if st.session_state['negative_prompt'] != st.session_state['negative_prompt_value']:
                    st.session_state['negative_prompt_value'] = st.session_state['negative_prompt']
                    st.experimental_rerun()
                st.session_state['guidance_scale'] = st.number_input(f"Guidance scale", value=float(st.session_state['guidance_scale']))
                st.session_state['seed'] = st.number_input(f"Seed", value=int(st.session_state['seed']))
                st.session_state['num_inference_steps'] = st.number_input(f"Inference steps", value=int(st.session_state['num_inference_steps']))
                            
        batch_run_range = st.sidebar.slider("Select range:", 1, 0, (0, len(timing_details)-1))  
        
        st.session_state["promote_new_generation"] = True                    
        st.session_state["promote_new_generation"] = st.sidebar.checkbox("Promote new generation to main variant", value=True, key="promote_new_generation_to_main_variant")

        app_settings = get_app_settings()

        if 'restyle_button' not in st.session_state:
            st.session_state['restyle_button'] = ''
            st.session_state['item_to_restyle'] = ''                

        btn1, btn2 = st.sidebar.columns(2)

        with btn1:
            batch_number_of_variants = st.number_input("How many variants?", value=1, min_value=1, max_value=10, step=1, key="number_of_variants")
        

        with btn2:

            st.write("")
            st.write("")
            if st.button(f'Batch restyle') or st.session_state['restyle_button'] == 'yes':
                                
                if st.session_state['restyle_button'] == 'yes':
                    range_start = int(st.session_state['item_to_restyle'])
                    range_end = range_start + 1
                    st.session_state['restyle_button'] = ''
                    st.session_state['item_to_restyle'] = ''

                for i in range(batch_run_range[1]+1):
                    for number in range(0, batch_number_of_variants):
                        index_of_current_item = i
                        trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],custom_models,st.session_state['adapter_type'])
                st.experimental_rerun()

        if st.session_state['frame_styling_view_type'] == "Single Frame":
        
            detail1, detail2, detail3, detail4 = st.columns([2,2,1,2])

            with detail1:
                individual_number_of_variants = st.number_input(f"How many variants?", min_value=1, max_value=10, value=1, key=f"number_of_variants_{st.session_state['which_image']}")
                
                
            with detail2:
                st.write("")
                st.write("")
                if st.button(f"Generate Variants", key=f"new_variations_{st.session_state['which_image']}",help="This will generate new variants based on the settings to the left."):
                    for i in range(0, individual_number_of_variants):
                        index_of_current_item = st.session_state['which_image']
                        trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],custom_models,st.session_state['adapter_type']) 
                    st.experimental_rerun()
                    
            with detail4:
                st.write("")
                with st.expander("💡 Editing key frames"):
                    st.info("You can edit the key frames in Tools > Frame Editing.")

        
        elif st.session_state['frame_styling_view_type'] == "List View":
            for i in range(0, len(timing_details)):
                index_of_current_item = i
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.subheader(f"Frame {i}")
                col2.empty()
                with col3:
                    if st.button("Delete this keyframe", key=f'{index_of_current_item}'):
                        delete_frame(project_name, index_of_current_item)
                        timing_details = get_timing_details(project_name)
                        st.experimental_rerun()           
                                    
                if timing_details[i]["alternative_images"] != "":
                    variants = timing_details[i]["alternative_images"]
                    current_variant = int(timing_details[i]["primary_image"])    
                    st.image(variants[current_variant])                            
                else:
                    st.image('https://i.ibb.co/GHVfjP0/Image-Not-Yet-Created.png', use_column_width=True) 
                

                detail1, detail2, detail3, detail4 = st.columns([2,2,1,3])

                with detail1:
                    individual_number_of_variants = st.number_input(f"How many variants?", min_value=1, max_value=10, value=1, key=f"number_of_variants_{index_of_current_item}")
                    

                    
                with detail2:
                    st.write("")
                    st.write("")
                    if st.button(f"Generate Variants", key=f"new_variations_{index_of_current_item}",help="This will generate new variants based on the settings to the left."):
                        for a in range(0, individual_number_of_variants):
                            index_of_current_item = i
                            trigger_restyling_process(timing_details, project_name, index_of_current_item,st.session_state['model'],st.session_state['prompt'],st.session_state['strength'],st.session_state['custom_pipeline'],st.session_state['negative_prompt'],st.session_state['guidance_scale'],st.session_state['seed'],st.session_state['num_inference_steps'],st.session_state['which_stage_to_run_on'],st.session_state["promote_new_generation"], st.session_state['project_settings'],custom_models,st.session_state['adapter_type'])                             
                        st.experimental_rerun()
                    
                    
                with detail3:
                    st.write("")
                with detail4:
                    if st.button(f"Jump to single frame view for #{index_of_current_item}", help="This will switch to a Single Frame view type and open this individual image."):
                        st.session_state['which_image_value'] = index_of_current_item
                        st.session_state['frame_styling_view_type'] = "Single View"
                        st.session_state['frame_styling_view_type_index'] = 1                                    
                        st.experimental_rerun() 
            
            
                
        st.markdown("***")        
        st.subheader("Preview video")
        st.write("You can get a gif of the video by clicking the button below.")
        if st.button("Create gif of current main variants"):
            
            create_gif_preview(project_name, timing_details)
            st.image(f"videos/{project_name}/preview_gif.gif", use_column_width=True)                
            st.balloons()
                                            
