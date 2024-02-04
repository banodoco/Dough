import json
import streamlit as st
from ui_components.methods.common_methods import process_inference_output,add_new_shot, save_uploaded_image
from ui_components.methods.file_methods import generate_pil_image
from ui_components.methods.ml_methods import query_llama2
from ui_components.widgets.add_key_frame_element import add_key_frame
from utils.common_utils import refresh_app
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from shared.constants import QUEUE_INFERENCE_QUERIES, AIModelType, InferenceType, InternalFileTag, InternalFileType, SortOrder
from utils import st_memory
import time
from utils.enum import ExtendedEnum
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.constants import ML_MODEL
from PIL import Image, ImageFilter
import io
import cv2
import numpy as np
from utils import st_memory


class InputImageStyling(ExtendedEnum):
    TEXT2IMAGE = "Text to Image"
    IMAGE2IMAGE = "Image to Image"
    CONTROLNET_CANNY = "ControlNet Canny"
    IPADAPTER_FACE = "IP-Adapter Face"
    IPADAPTER_PLUS = "IP-Adapter Plus"
    IPADPTER_FACE_AND_PLUS = "IP-Adapter Face & Plus"


def explorer_page(project_uuid):
    data_repo = DataRepo()
    project_setting = data_repo.get_project_setting(project_uuid)        

    st.markdown(f"#### :red[{st.session_state['main_view_type']}] > :green[{st.session_state['page']}]")

    st.markdown("***")
        
    with st.expander("✨ Generate Images", expanded=True):
        generate_images_element(position='explorer', project_uuid=project_uuid, timing_uuid=None)
    st.markdown("***")

        
    gallery_image_view(project_uuid,False,view=['add_and_remove_from_shortlist','view_inference_details'])




def generate_images_element(position='explorer', project_uuid=None, timing_uuid=None):
    data_repo = DataRepo()
    project_settings = data_repo.get_project_setting(project_uuid)
    help_input='''This will generate a specific prompt based on your input.\n\n For example, "Sad scene of old Russian man, dreary style" might result in "Boris Karloff, 80 year old man wearing a suit, standing at funeral, dark blue watercolour."'''
    a1, a2, a3 = st.columns([1,1,0.3])   
    

    with a1 if 'switch_prompt_position' not in st.session_state or st.session_state['switch_prompt_position'] == False else a2:
        prompt = st_memory.text_area("What's your base prompt?", key="explorer_base_prompt", help="This exact text will be included for each generation.")

    with a2 if 'switch_prompt_position' not in st.session_state or st.session_state['switch_prompt_position'] == False else a1:
        magic_prompt = st_memory.text_area("What's your magic prompt?", key="explorer_magic_prompt", help=help_input)
        #if magic_prompt != "":
        #    chaos_level = st_memory.slider("How much chaos would you like to add to the magic prompt?", min_value=0, max_value=100, value=20, step=1, key="chaos_level", help="This will determine how random the generated prompt will be.")                    
        #    temperature = chaos_level / 20
        temperature = 1.0
    with a3:
        st.write("")
        st.write("")
        st.write("")
        if st.button("🔄", key="switch_prompt_position_button:", use_container_width=True, help="This will switch the order the prompt and magic prompt are used - earlier items gets more attention."):
            st.session_state['switch_prompt_position'] = not st.session_state.get('switch_prompt_position', False)
            st.experimental_rerun()

    neg1, _ = st.columns([1,1.3])
    with neg1:
        negative_prompt = st_memory.text_input("Negative prompt:", value="bad image, worst image, bad anatomy, washed out colors",\
                                            key="explorer_neg_prompt", \
                                                help="These are the things you wish to be excluded from the image")

    b1, b2, b3,b4 = st.columns([1.5,1.5,1.5,1])
    c1, c2, _ = st.columns([2,2,2])
                     

    with b1:
        type_of_generation = st_memory.radio("How would you like to generate the image?", options=InputImageStyling.value_list(), key="type_of_generation_key", help="Evolve Image will evolve the image based on the prompt, while Maintain Structure will keep the structure of the image and change the style.",horizontal=True) 
        
        
    input_image_key = "input_image_1"
    if input_image_key not in st.session_state:
        st.session_state[input_image_key] = None
    if 'input_image_2' not in st.session_state:
        st.session_state['input_image_2'] = None
    if type_of_generation != InputImageStyling.TEXT2IMAGE.value:
        with b2:
            source_of_starting_image = st_memory.radio("Image source:", options=["Upload", "From Shot"], key="source_of_starting_image", help="This will be the base image for the generation.",horizontal=True)
            if source_of_starting_image == "Upload":
                input_image = st.file_uploader("Upload a starting image", type=["png", "jpg", "jpeg"], key="explorer_input_image", help="This will be the base image for the generation.")                                        
            else:
                shot_list = data_repo.get_shot_list(project_uuid)
                selection1, selection2 = st.columns([1,1])
                with selection1:
                    shot_name = st.selectbox("Shot:", options=[shot.name for shot in shot_list], key="explorer_shot_uuid", help="This will be the base image for the generation.")
                shot_uuid = [shot.uuid for shot in shot_list if shot.name == shot_name][0]
                frame_list = data_repo.get_timing_list_from_shot(shot_uuid)
                with selection2:
                    list_of_timings = [i + 1 for i in range(len(frame_list))]
                    timing = st.selectbox("Frame #:", options=list_of_timings, key="explorer_frame_number", help="This will be the base image for the generation.")
                    #timing = st.number_input("Frame #:", min_value=1, max_value=len(frame_list), value=1, step=1, key="explorer_frame_number", help="This will be the base image for the generation.")
                input_image = frame_list[timing - 1].primary_image.location
                # make it a byte stream
                st.image(frame_list[timing - 1].primary_image.location, use_column_width=True)
                
            if type_of_generation == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                source_of_starting_image_2 = st_memory.radio("How would you like to upload the second starting image?", options=["Upload", "From Shot"], key="source_of_starting_image_2", help="This will be the base image for the generation.",horizontal=True)
                if source_of_starting_image_2 == "Upload":
                    input_image_2 = st.file_uploader("IP-Adapter Face image:", type=["png", "jpg", "jpeg"], key="explorer_input_image_2", help="This will be the base image for the generation.")
                else:
                    selection1, selection2 = st.columns([1,1])
                    with selection1:   
                        shot_list = data_repo.get_shot_list(project_uuid)
                        shot_name = st.selectbox("Shot:", options=[shot.name for shot in shot_list], key="explorer_shot_uuid_2", help="This will be the base image for the generation.")
                        shot_uuid = [shot.uuid for shot in shot_list if shot.name == shot_name][0]
                    with selection2:
                        frame_list = data_repo.get_timing_list_from_shot(shot_uuid)
                        list_of_timings = [i + 1 for i in range(len(frame_list))]
                        timing = st.selectbox("Frame #:", options=list_of_timings, key="explorer_frame_number_2", help="This will be the base image for the generation.")
                    input_image_2 = frame_list[timing - 1].primary_image.location
                    st.image(frame_list[timing - 1].primary_image.location, use_column_width=True)

            # if type type is face and plus, then we need to make the text images
            if type_of_generation == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                button_text = "Upload Images"
            else:
                button_text = "Upload Image"
            if st.button(button_text, use_container_width=True):                                                
                st.session_state[input_image_key] = input_image   
                if type_of_generation == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                    st.session_state['input_image_2'] = input_image_2   
                st.rerun()      
        with b3:
            # prompt_strength = round(1 - (strength_of_image / 100), 2)
            if type_of_generation != InputImageStyling.IPADPTER_FACE_AND_PLUS.value:                                             
                if st.session_state[input_image_key] is not None:
                    st.info("Current image:")                                
                    st.image(st.session_state[input_image_key], use_column_width=True)
                else:
                    st.info("Current image:")      
                    st.error("Please upload an image")

        with b3:
            edge_pil_img = None
            # strength_of_image = st_memory.slider("What % of the current image would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_image_key", help="This will determine how much of the current image will be kept in the final image.")                    
        if type_of_generation == InputImageStyling.IMAGE2IMAGE.value:      
            with b3:                                      
                strength_of_image = st_memory.slider("How much blur would you like to add to the image?", min_value=0, max_value=100, value=50, step=1, key="strength_of_image2image", help="This will determine how much of the current image will be kept in the final image.")


        elif type_of_generation == InputImageStyling.CONTROLNET_CANNY.value:   
            with b3:
                strength_of_image = st_memory.slider("How much of the current image would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_controlnet_canny", help="This will determine how much of the current image will be kept in the final image.")                     

        elif type_of_generation == InputImageStyling.IPADAPTER_FACE.value:
            with b3:
                strength_of_image = st_memory.slider("How much of the current image would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_ipadapter_face", help="This will determine how much of the current image will be kept in the final image.")

        elif type_of_generation == InputImageStyling.IPADAPTER_PLUS.value:
            with b3:                
                strength_of_plus = st_memory.slider("How much of the current image would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_ipadapter_plus", help="This will determine how much of the current image will be kept in the final image.")                            

        elif type_of_generation == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
            with b3:
                if st.session_state[input_image_key] is not None:  
                    st.info("IP-Adapter Face image:")  
                    st.image(st.session_state[input_image_key], use_column_width=True)
                    strength_of_face = st_memory.slider("How strong would would you like the Face model to influence?", min_value=0, max_value=100, value=50, step=1, key="strength_of_ipadapter_face", help="This will determine how much of the current image will be kept in the final image.")
                else:
                    st.info("IP-Adapter Face image:")
                    st.error("Please upload an image")
                if st.session_state['input_image_2'] is not None:  
                    st.info("IP-Adapter Plus image:")  
                    st.image(st.session_state['input_image_2'], use_column_width=True)
                    strength_of_plus = st_memory.slider("How strong would you like to influence the Plus model?", min_value=0, max_value=100, value=50, step=1, key="strength_of_ipadapter_plus", help="This will determine how much of the current image will be kept in the final image.")
                else:
                    st.info("IP-Adapter Plus image:")
                    st.error("Please upload an second image")
                
    if type_of_generation != InputImageStyling.TEXT2IMAGE.value:
        
        if st.session_state[input_image_key] is not None:
            with b3:
                if st.button("Clear input image", key="clear_input_image", use_container_width=True):
                    st.session_state[input_image_key] = None
                    st.session_state['input_image_2'] = None
                    st.rerun()
            
    if not st.session_state[input_image_key]:
        input_image = None
        type_of_generation = None
        strength_of_image = None
    # st.markdown("***")
    model_name = "stable_diffusion_xl"
    if position=='explorer':
        _, d2,d3, _ = st.columns([0.25, 1,1, 0.25])
    else:
        d2,d3 = st.columns([1,1])
    with d2:        
        number_to_generate = st.slider("How many images would you like to generate?", min_value=0, max_value=100, value=4, step=4, key="number_to_generate", help="It'll generate 4 from each variation.")
    
    with d3:
        st.write(" ")                
        if st.session_state.get(position + '_generate_inference'):
            ml_client = get_ml_client()
            counter = 0
            for _ in range(number_to_generate):
                
                if counter % 4 == 0:
                    if magic_prompt != "":
                        input_text = "I want to flesh the following user input out - could you make it such that it retains the original meaning but is more specific and descriptive:\n\nfloral background|array of colorful wildflowers and green foliage forms a vibrant, natural backdrop.\nfancy old man|Barnaby Jasper Hawthorne, a dignified gentleman in his late seventies\ncomic book style|illustration style of a 1960s superhero comic book\nsky with diamonds|night sky filled with twinkling stars like diamonds on velvet\n20 y/o indian guy|Piyush Ahuja, a twenty-year-old Indian software engineer\ndark fantasy|a dark, gothic style similar to an Edgar Allen Poe novel\nfuturistic world|set in a 22nd century off-world colony called Ajita Iyera\nbeautiful lake|the crystal clear waters of a luminous blue alpine mountain lake\nminimalistic illustration|simple illustration with solid colors and basic geometrical shapes and figures\nmale blacksmith|Arun Thakkar, a Black country village blacksmith\ndesert sunrise|reddish orange sky at sunrise somewhere out in the Arabia desert\nforest|dense forest of Swedish pine trees\ngreece landscape|bright cyan sky meets turquoise on Santorini\nspace|shifting nebula clouds across the endless expanse of deep space\nwizard orcs|Poljak Ardell, a half-orc warlock\ntropical island|Palm tree-lined tropical paradise beach near Corfu\ncyberpunk cityscape  |Neon holo displays reflect from steel surfaces of buildings in Cairo Cyberspace\njapanese garden & pond|peaceful asian zen koi fishpond surrounded by bonsai trees\nattractive young african woman|Chimene Nkasa, young Congolese social media star\ninsane style|wild and unpredictable artwork like Salvador Dali’s Persistence Of Memory painting\n30s european women|Francisca Sampere, 31 year old Spanish woman\nlighthouse|iconic green New England coastal lighthouse against grey sky\ngirl in hat|Dora Alamanni dressed up with straw boater hat\nretro poster design|stunning vintage 80s movie poster reminiscent of Blade Runner\nabstract color combinations|a modernist splatter painting with overlapping colors\nnordic style |simple line drawing of white on dark blue with clean geometrical figures and shapes\nyoung asian woman, abstract style|Kaya Suzuki's face rendered in bright, expressive brush strokes\nblue monster|large cobalt blue cartoonish creature similar to a yeti\nman at work|portrait sketch of business man working late night in the office\nunderwater sunbeams|aquatic creatures swimming through waves of refracting ocean sunlight\nhappy cat on table|tabby kitten sitting alert in anticipation on kitchen counter\ntop​\nold timey train robber|Wiley Hollister, mid-thirties outlaw\nchinese landscape|Mt. Taihang surrounded by clouds\nancient ruins, sci fi style|deserted ancient civilization under stormy ominous sky full of mysterious UFOs\nanime art|classic anime, in the style of Akira Toriyama\nold man, sad scene|Seneca Hawkins, older gentleman slumped forlorn on street bench in early autumn evening\ncathedral|interior view of Gothic church in Vienna\ndreamlike|spellbinding dreamlike atmosphere, work called Pookanaut\nbird on lake, evening time|grizzled kingfisher sitting regally facing towards beautiful ripple-reflected setting orange pink sum\nyoung female character, cutsey style|Aoife Delaney dressed up as Candyflud, cheerful child adventurer\ninteresting style|stunning cubist abstract geometrical block\nevil woman|Luisa Schultze, frightening murderess\nfashion model|Ishita Chaudry, an Indian fashionista with unique dress sense\ncastle, moody scene|grand Renaissance Palace in Prague against twilight mist filled with crows\ntropical paradise island|Pristine white sand beach with palm trees at Ile du Mariasi, Reunion\npoverty stricken village|simple shack-based settlement in rural Niger\ngothic horror creature|wretchedly deformed and hideous tatter-clad creature like Caliban from Shakespeare ’s Tempes\nlots of color|rainbow colored Dutch flower field\nattractive woman on holidays|Siena Chen in her best little black dress, walking down a glamorous Las Vegas Boulevard\nItalian city scene|Duomo di Milano on dark rainy night sky behind it\nhappy dog outdoor|bouncy Irish Setter frolickling around green grass in summer sun\nmedieval fantasy world|illustration work for Eye Of The Titan - novel by Rania D’Allara\nperson relaxing|Alejandro Gonzalez sitting crosslegged in elegant peacock blue kurta while reading book\nretro sci fi robot|Vintage, cartoonish android reminiscent of the Bender Futurama character. Named Clyde Frost.\ngeometric style|geometric abstract style based on 1960 Russian poster design by Alexander Rodchenk \nbeautiful girl face, vaporwave style|Rayna Vratasky, looking all pink and purple retro\nspooking |horrifying Chupacabra-like being staring intensely to camera\nbrazilian woman having fun|Analia Santos, playing puzzle game with friends\nfemale elf warrior|Finnula Thalas, an Eladrin paladin wielding two great warblades\nlsd trip scene|kaleidoscopic colorscape, filled with ephemerally shifting forms\nyoung african man headshot|Roger Mwafulo looking sharp with big lush smile\nsad or dying person|elderly beggar Jeon Hagopian slumped against trash can bin corner\nart |neurologically inspired psychedelian artwork like David Normal's “Sentient Energy ” series\nattractive german woman|Johanna Hecker, blonde beauty with long hair wrapped in braid ties\nladybug|Cute ladybug perched on red sunset flower petals on summery meadow backdrop\nbeautiful asian women |Chiraya Phetlue, Thai-French model standing front view wearing white dress\nmindblowing style|trippy space illustration that could be cover for a book by Koyu Azumi\nmoody|forest full of thorn trees stretching into the horizon at dusk\nhappy family, abstract style|illustration work of mother, father and child from 2017 children’s picture book The Gifts Of Motherhood By Michelle Sparks\n"
                        output_magic_prompt=query_llama2(f"{input_text}{magic_prompt}|", temperature=temperature)
                    else:
                        output_magic_prompt = ""                                        
                if 'switch_prompt_position' not in st.session_state or st.session_state['switch_prompt_position'] == False:                
                    prompt_with_variations = f"{prompt}, {output_magic_prompt}" if prompt else output_magic_prompt
                    
                else:  # switch_prompt_position is True
                    prompt_with_variations = f"{output_magic_prompt}, {prompt}" if prompt else output_magic_prompt
                    

                counter += 1
                log = None

                
                if InputImageStyling.value_list()[st.session_state['type_of_generation_key']] == InputImageStyling.TEXT2IMAGE.value:
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        guidance_scale=5,
                        seed=-1,                            
                        num_inference_steps=30,            
                        strength=1,
                        adapter_type=None,
                        prompt=prompt_with_variations,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid
                    )

                    model_list = data_repo.get_all_ai_model_list(model_type_list=[AIModelType.TXT2IMG.value], custom_trained=False)
                    model_dict = {}
                    for m in model_list:
                        model_dict[m.name] = m

                    replicate_model = ML_MODEL.get_model_by_db_obj(model_dict[model_name])
                    output, log = ml_client.predict_model_output_standardized(replicate_model, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                else:
                    if InputImageStyling.value_list()[st.session_state['type_of_generation_key']] == InputImageStyling.IMAGE2IMAGE.value:
                        input_image_file = save_uploaded_image(input_image, project_uuid)
                        query_obj = MLQueryObject(
                            timing_uuid=None,
                            model_uuid=None,
                            image_uuid=input_image_file.uuid,
                            guidance_scale=5,
                            seed=-1,
                            num_inference_steps=30,
                            strength=prompt_strength,
                            adapter_type=None,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=project_settings.height,
                            width=project_settings.width,
                            project_uuid=project_uuid
                        )

                        output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                    elif InputImageStyling.value_list()[st.session_state['type_of_generation_key']] == InputImageStyling.CONTROLNET_CANNY.value:
                        input_image_file = save_uploaded_image(edge_pil_img, project_uuid)
                        query_obj = MLQueryObject(
                            timing_uuid=None,
                            model_uuid=None,
                            image_uuid=input_image_file.uuid,
                            guidance_scale=5,
                            seed=-1,
                            num_inference_steps=30,
                            strength=0.5,
                            adapter_type=None,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=project_settings.height,
                            width=project_settings.width,
                            project_uuid=project_uuid,
                            data={'condition_scale': condition_scale}
                        )

                        output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl_controlnet, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)
                    
                    elif InputImageStyling.value_list()[st.session_state['type_of_generation_key']] == InputImageStyling.IPADAPTER_FACE.value:
                        st.write("Not implemented yet")
                    
                    elif InputImageStyling.value_list()[st.session_state['type_of_generation_key']] == InputImageStyling.IPADAPTER_PLUS.value:
                        st.write("Not implemented yet")

                    elif InputImageStyling.value_list()[st.session_state['type_of_generation_key']] == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                        st.write("Not implemented yet")

                if log:
                    inference_data = {
                        "inference_type": InferenceType.GALLERY_IMAGE_GENERATION.value if position == 'explorer' else InferenceType.FRAME_TIMING_IMAGE_INFERENCE.value,
                        "output": output,
                        "log_uuid": log.uuid,
                        "project_uuid": project_uuid,
                        "timing_uuid": timing_uuid,
                        "promote_new_generation": False
                    }
                    process_inference_output(**inference_data)

            st.info("Check the Generation Log to the left for the status.")
            time.sleep(0.5)
            toggle_generate_inference(position)
            st.rerun()

        # ----------- generate btn --------------
        st.button("Generate images", key="generate_images", use_container_width=True, type="primary", on_click=lambda: toggle_generate_inference(position))
            
def toggle_generate_inference(position):
    if position + '_generate_inference' not in st.session_state:
        st.session_state[position + '_generate_inference'] = True
    else:
        st.session_state[position + '_generate_inference'] = not st.session_state[position + '_generate_inference']


def gallery_image_view(project_uuid, shortlist=False, view=["main"], shot=None, sidebar=False):
    data_repo = DataRepo()
    
    project_settings = data_repo.get_project_setting(project_uuid)
    shot_list = data_repo.get_shot_list(project_uuid)
    k1,k2 = st.columns([5,1])
    if sidebar != True:
        f1, f2 = st.columns([1, 1])
        with f1:
            num_columns = st_memory.slider('Number of columns:', min_value=3, max_value=7, value=4,key="num_columns_explorer")
        with f2:
            num_items_per_page = st_memory.slider('Items per page:', min_value=10, max_value=50, value=16, key="num_items_per_page_explorer")

        if shortlist is False:
            page_number = k1.radio("Select page:", options=range(1, project_settings.total_gallery_pages + 1), horizontal=True, key="main_gallery")
            if 'view_inference_details' in view:
                open_detailed_view_for_all = k2.toggle("Open detailed view for all:", key='main_gallery_toggle')
            st.markdown("***")
        else:
            project_setting = data_repo.get_project_setting(project_uuid)
            page_number = k1.radio("Select page", options=range(1, project_setting.total_shortlist_gallery_pages), horizontal=True, key="shortlist_gallery")
            open_detailed_view_for_all = False     
            st.markdown("***")
            
            

    else:
        project_setting = data_repo.get_project_setting(project_uuid)
        page_number = k1.radio("Select page", options=range(1, project_setting.total_shortlist_gallery_pages), horizontal=True, key="shortlist_gallery")
        open_detailed_view_for_all = False        
        num_items_per_page = 8
        num_columns = 2
    
    gallery_image_list, res_payload = data_repo.get_all_file_list(
        file_type=InternalFileType.IMAGE.value, 
        tag=InternalFileTag.GALLERY_IMAGE.value if not shortlist else InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value, 
        project_id=project_uuid,
        page=page_number or 1,
        data_per_page=num_items_per_page,
        sort_order=SortOrder.DESCENDING.value 
    )

    if not shortlist:
        if project_settings.total_gallery_pages != res_payload['total_pages']:
            project_settings.total_gallery_pages = res_payload['total_pages']
            st.rerun()
    else:
        if project_settings.total_shortlist_gallery_pages != res_payload['total_pages']:
            project_settings.total_shortlist_gallery_pages = res_payload['total_pages']
            st.rerun()

    # def is_image_truncated(image_path):
    #     try:
    #         img = Image.open(image_path)
    #         img.verify()  # verify that it is, in fact an image
    #     except (IOError, SyntaxError) as e:
    #         return True
    #     return False
    if shortlist is False:
        fetch1, fetch2, fetch3, fetch4 = st.columns([0.25, 1, 1, 0.25])
        st.markdown("***")
        with fetch2:
            st.info("###### 25 images pending")     
        with fetch3:
            image_pending = 8
            if image_pending:                            
                if st.button("Check for new images", key=f"check_for_new_images_", use_container_width=True):
                    st.write("Fetching images...")
                # st.markdown("***")
    total_image_count = res_payload['count']
    if gallery_image_list and len(gallery_image_list):
        start_index = 0
        end_index = min(start_index + num_items_per_page, total_image_count)
        shot_names = [s.name for s in shot_list]
        shot_names.append('**Create New Shot**')        
        for i in range(start_index, end_index, num_columns):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                if i + j < len(gallery_image_list):
                    with cols[j]:                        
                        st.image(gallery_image_list[i + j].location, use_column_width=True)
                        # else:
                        #     st.error("The image is truncated and cannot be displayed.")
                        if 'add_and_remove_from_shortlist' in view:
                            if shortlist:
                                if st.button("Remove from shortlist ➖", key=f"shortlist_{gallery_image_list[i + j].uuid}",use_container_width=True, help="Remove from shortlist"):
                                    data_repo.update_file(gallery_image_list[i + j].uuid, tag=InternalFileTag.GALLERY_IMAGE.value)
                                    st.success("Removed From Shortlist")
                                    time.sleep(0.3)
                                    st.rerun()
                            else:
                                if st.button("Add to shortlist ➕", key=f"shortlist_{gallery_image_list[i + j].uuid}",use_container_width=True, help="Add to shortlist"):
                                    data_repo.update_file(gallery_image_list[i + j].uuid, tag=InternalFileTag.SHORTLISTED_GALLERY_IMAGE.value)
                                    st.success("Added To Shortlist")
                                    time.sleep(0.3)
                                    st.rerun()

                        # -------- inference details --------------          
                        if gallery_image_list[i + j].inference_log:
                            log = gallery_image_list[i + j].inference_log # data_repo.get_inference_log_from_uuid(gallery_image_list[i + j].inference_log.uuid)
                            if log:
                                input_params = json.loads(log.input_params)
                                prompt = input_params.get('prompt', 'No prompt found')
                                model = json.loads(log.output_details)['model_name'].split('/')[-1]
                                if 'view_inference_details' in view:
                                    with st.expander("Prompt Details", expanded=open_detailed_view_for_all):
                                        st.info(f"**Prompt:** {prompt}\n\n**Model:** {model}")
                                
                            else:
                                st.warning("No inference data")
                        else:
                            st.warning("No data found")

                        # ---------- add to shot btn ---------------
                        if "last_shot_number" not in st.session_state:
                            st.session_state["last_shot_number"] = 0
                        if 'add_to_this_shot' in view or 'add_to_any_shot' in view:
                            if 'add_to_this_shot' in view:
                                shot_name = shot.name
                            else:
                                shot_name = st.selectbox('Add to shot:', shot_names, key=f"current_shot_sidebar_selector_{gallery_image_list[i + j].uuid}",index=st.session_state["last_shot_number"])
                            
                            if shot_name != "":
                                if shot_name == "**Create New Shot**":
                                    shot_name = st.text_input("New shot name:", max_chars=40, key=f"shot_name_{gallery_image_list[i+j].uuid}")
                                    if st.button("Create new shot", key=f"create_new_{gallery_image_list[i + j].uuid}", use_container_width=True):
                                        new_shot = add_new_shot(project_uuid, name=shot_name)
                                        add_key_frame(gallery_image_list[i + j], False, new_shot.uuid, len(data_repo.get_timing_list_from_shot(new_shot.uuid)), refresh_state=False)
                                        # removing this from the gallery view
                                        data_repo.update_file(gallery_image_list[i + j].uuid, tag="")
                                        st.rerun()
                                    
                                else:
                                    if st.button(f"Add to shot", key=f"add_{gallery_image_list[i + j].uuid}", help="Promote this variant to the primary image", use_container_width=True):
                                        shot_number = shot_names.index(shot_name)
                                        st.session_state["last_shot_number"] = shot_number 
                                        shot_uuid = shot_list[shot_number].uuid

                                        add_key_frame(gallery_image_list[i + j], False, shot_uuid, len(data_repo.get_timing_list_from_shot(shot_uuid)), refresh_state=False)
                                        # removing this from the gallery view
                                        data_repo.update_file(gallery_image_list[i + j].uuid, tag="")
                                        refresh_app(maintain_state=True)
                                                    
            st.markdown("***")
    else:
        st.warning("No images present")



'''

def update_max_frame_per_shot_element(project_uuid):
    data_repo = DataRepo()
    project_settings = data_repo.get_project_setting(project_uuid)

    
    max_frames = st.number_input(label='Max frames per shot', min_value=1, value=project_settings.max_frames_per_shot)

    if max_frames != project_settings.max_frames_per_shot:
        project_settings.max_frames_per_shot = max_frames
        st.success("Updated")
        time.sleep(0.3)
        st.rerun()
'''