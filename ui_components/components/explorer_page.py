import json
import streamlit as st
from ui_components.methods.common_methods import get_canny_img, process_inference_output,add_new_shot, save_new_image, save_uploaded_image
from ui_components.methods.file_methods import generate_pil_image
from ui_components.methods.ml_methods import query_llama2
from ui_components.widgets.add_key_frame_element import add_key_frame
from utils.common_utils import refresh_app
from utils.constants import MLQueryObject
from utils.data_repo.data_repo import DataRepo
from shared.constants import GPU_INFERENCE_ENABLED, QUEUE_INFERENCE_QUERIES, AIModelType, InferenceType, InternalFileTag, InternalFileType, SortOrder
from utils import st_memory
import time
from utils.enum import ExtendedEnum
from utils.ml_processor.ml_interface import get_ml_client
from utils.ml_processor.constants import ML_MODEL
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
       negative_prompt = st_memory.text_area("Negative prompt:", value="bad image, worst image, bad anatomy, washed out colors",\
                                            key="explorer_neg_prompt", \
                                                help="These are the things you wish to be excluded from the image")


    b1, b2, b3, _ = st.columns([1.5,1,1.5,1])
    with b1:
        type_of_generation = st_memory.radio("How would you like to generate the image?", options=InputImageStyling.value_list(), key="type_of_generation_key", help="Evolve Image will evolve the image based on the prompt, while Maintain Structure will keep the structure of the image and change the style.",horizontal=True) 

    input_image_1_key = "input_image_1"
    input_image_2_key = "input_image_2"
    if input_image_1_key not in st.session_state:
        st.session_state[input_image_1_key] = None
        st.session_state[input_image_2_key] = None

    uploaded_image_1 = None
    uploaded_image_2 = None

    # these require two images
    ipadapter_types = [InputImageStyling.IPADPTER_FACE_AND_PLUS.value]

    # UI for image input if type_of_generation is not txt2img
    if type_of_generation != InputImageStyling.TEXT2IMAGE.value:
        # UI - Base Input
        with b2:
            source_of_starting_image = st_memory.radio("Image source:", options=["Upload", "From Shot"], key="source_of_starting_image", help="This will be the base image for the generation.",horizontal=True)
            # image upload
            if source_of_starting_image == "Upload":
                uploaded_image_1 = st.file_uploader("Upload a starting image", type=["png", "jpg", "jpeg"], key="explorer_input_image", help="This will be the base image for the generation.")
            # taking image from shots
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

                uploaded_image_1 = frame_list[timing - 1].primary_image.location
                # make it a byte stream
                st.image(frame_list[timing - 1].primary_image.location, use_column_width=True)
            
            # taking a second image in the case of ip_adapter_face_plus
            if type_of_generation in ipadapter_types:
                source_of_starting_image_2 = st_memory.radio("How would you like to upload the second starting image?", options=["Upload", "From Shot"], key="source_of_starting_image_2", help="This will be the base image for the generation.",horizontal=True)
                if source_of_starting_image_2 == "Upload":
                    uploaded_image_2 = st.file_uploader("IP-Adapter Face image:", type=["png", "jpg", "jpeg"], key="explorer_input_image_2", help="This will be the base image for the generation.")
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
                    uploaded_image_2 = frame_list[timing - 1].primary_image.location
                    st.image(frame_list[timing - 1].primary_image.location, use_column_width=True)

            # if type type is face and plus, then we need to make the text images
            button_text = "Upload Images" if type_of_generation in ipadapter_types else "Upload Image"

            if st.button(button_text, use_container_width=True):                                                
                st.session_state[input_image_1_key] = uploaded_image_1   
                st.session_state[input_image_2_key] = uploaded_image_2   
                st.rerun()

        # UI - Preview
        with b3:
            # prompt_strength = round(1 - (strength_of_image / 100), 2)
            if type_of_generation not in ipadapter_types:                                             
                st.info("Current image:")
                if st.session_state[input_image_1_key] is not None:
                    st.image(st.session_state[input_image_1_key], use_column_width=True)
                else:
                    st.error("Please upload an image")
        
            if type_of_generation == InputImageStyling.IMAGE2IMAGE.value:                                                 
                strength_of_image = st_memory.slider("How much blur would you like to add to the image?", min_value=0, max_value=100, value=50, step=1, key="strength_of_image2image", help="This will determine how much of the current image will be kept in the final image.")

            elif type_of_generation == InputImageStyling.CONTROLNET_CANNY.value:                
                strength_of_image = st_memory.slider("How much of the current image would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_controlnet_canny", help="This will determine how much of the current image will be kept in the final image.")

            elif type_of_generation == InputImageStyling.IPADAPTER_FACE.value:                
                strength_of_image = st_memory.slider("How much of the current image would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_ipadapter_face", help="This will determine how much of the current image will be kept in the final image.")

            elif type_of_generation == InputImageStyling.IPADAPTER_PLUS.value:                
                strength_of_image = st_memory.slider("How much of the current image would you like to keep?", min_value=0, max_value=100, value=50, step=1, key="strength_of_ipadapter_plus", help="This will determine how much of the current image will be kept in the final image.")                            

            elif type_of_generation == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                # UI - displaying uploaded images
                
                st.info("IP-Adapter Face image:")
                if st.session_state[input_image_1_key] is not None:    
                    st.image(st.session_state[input_image_1_key], use_column_width=True)
                    strength_of_face = st_memory.slider("How strong would would you like the Face model to influence?", min_value=0, max_value=100, value=50, step=1, key="strength_of_ipadapter_face", help="This will determine how much of the current image will be kept in the final image.")
                else:
                    st.error("Please upload an image")
                
                st.info("IP-Adapter Plus image:")
                if st.session_state[input_image_2_key] is not None:  
                    st.image(st.session_state[input_image_2_key], use_column_width=True)
                    strength_of_plus = st_memory.slider("How strong would you like to influence the Plus model?", min_value=0, max_value=100, value=50, step=1, key="strength_of_ipadapter_plus", help="This will determine how much of the current image will be kept in the final image.")
                else:
                    st.error("Please upload an second image")
        
        # UI - clear btn
        if st.session_state[input_image_1_key] is not None:
            with b3:
                if st.button("Clear input image(s)", key="clear_input_image", use_container_width=True):
                    st.session_state[input_image_1_key] = None
                    st.session_state[input_image_2_key] = None
                    st.rerun()

    if position == 'explorer':
        _, d2,d3, _ = st.columns([0.25, 1,1, 0.25])
    else:
        d2, d3 = st.columns([1,1])
    with d2:        
        number_to_generate = st.slider("How many images would you like to generate?", min_value=0, max_value=100, value=4, step=2, key="number_to_generate", help="It'll generate 4 from each variation.")
    
    with d3:
        st.write(" ")
        # ------------------- Generating output -------------------------------------
        if st.session_state.get(position + '_generate_inference'):
            ml_client = get_ml_client()
            counter = 0

            magic_prompt, temperature = "", 0
            for _ in range(number_to_generate):

                '''
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
                '''
                counter += 1
                log = None
                generation_method = InputImageStyling.value_list()[st.session_state['type_of_generation_key']]
                if generation_method == InputImageStyling.TEXT2IMAGE.value:
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        guidance_scale=8,
                        seed=-1,                            
                        num_inference_steps=25,            
                        strength=0.5,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid
                    )

                    # NOTE: code not is use
                    # model_list = data_repo.get_all_ai_model_list(model_type_list=[AIModelType.TXT2IMG.value], custom_trained=False)
                    # model_dict = {}
                    # for m in model_list:
                    #     model_dict[m.name] = m
                    # replicate_model = ML_MODEL.get_model_by_db_obj(model_dict[model_name])

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)
                
                elif generation_method == InputImageStyling.IMAGE2IMAGE.value:
                    input_image_file = save_new_image(st.session_state[input_image_1_key], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=input_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=0.8,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                elif generation_method == InputImageStyling.CONTROLNET_CANNY.value:
                    edge_pil_img = get_canny_img(st.session_state[input_image_1_key], low_threshold=50, high_threshold=150)    # redundant incase of local inference
                    input_img = edge_pil_img if not GPU_INFERENCE_ENABLED else st.session_state[input_image_1_key]
                    input_image_file = save_new_image(input_img, project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=input_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image/100,
                        adapter_type=None,
                        prompt=prompt,
                        low_threshold=0.3,
                        high_threshold=0.9,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={'condition_scale': 1}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.sdxl_controlnet, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)
                
                elif generation_method == InputImageStyling.IPADAPTER_FACE.value:
                    # validation
                    if not (st.session_state[input_image_1_key]):
                        st.error('Please upload an image')
                        return

                    input_image_file = save_new_image(st.session_state[input_image_1_key], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=input_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image/100,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.ipadapter_face, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)
                
                elif generation_method == InputImageStyling.IPADAPTER_PLUS.value:
                    input_image_file = save_new_image(st.session_state[input_image_1_key], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=input_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=strength_of_image/100,
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={'condition_scale': 1}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.ipadapter_plus, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

                elif generation_method == InputImageStyling.IPADPTER_FACE_AND_PLUS.value:
                    # validation
                    if not (st.session_state[input_image_2_key] and st.session_state[input_image_1_key]):
                        st.error('Please upload both images')
                        return

                    plus_image_file = save_new_image(st.session_state[input_image_1_key], project_uuid)
                    face_image_file = save_new_image(st.session_state[input_image_2_key], project_uuid)
                    query_obj = MLQueryObject(
                        timing_uuid=None,
                        model_uuid=None,
                        image_uuid=plus_image_file.uuid,
                        guidance_scale=5,
                        seed=-1,
                        num_inference_steps=30,
                        strength=(strength_of_face/100, strength_of_plus/100), # (face, plus)
                        adapter_type=None,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=project_settings.height,
                        width=project_settings.width,
                        project_uuid=project_uuid,
                        data={'file_image_2_uuid': face_image_file.uuid}
                    )

                    output, log = ml_client.predict_model_output_standardized(ML_MODEL.ipadapter_face_plus, query_obj, queue_inference=QUEUE_INFERENCE_QUERIES)

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
        if prompt == "":
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please enter a prompt to generate images")
        elif type_of_generation == InputImageStyling.IMAGE2IMAGE.value and st.session_state[input_image_1_key] is None:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload an image")
        elif type_of_generation == InputImageStyling.CONTROLNET_CANNY.value and st.session_state[input_image_1_key] is None:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload an image")
        elif type_of_generation == InputImageStyling.IPADAPTER_FACE.value and st.session_state[input_image_1_key] is None:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload an image")
        elif type_of_generation == InputImageStyling.IPADAPTER_PLUS.value and st.session_state[input_image_1_key] is None:
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload an image")
        elif type_of_generation == InputImageStyling.IPADPTER_FACE_AND_PLUS.value and (st.session_state[input_image_1_key] is None or st.session_state[input_image_2_key] is None):
            st.button("Generate images", key="generate_images", use_container_width=True, type="primary", disabled=True, help="Please upload both images")        
        else:
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

    if shortlist is False:
        _, fetch2, fetch3, _ = st.columns([0.25, 1, 1, 0.25])
        # st.markdown("***")
        explorer_stats = data_repo.get_explorer_pending_stats(project_uuid=project_uuid)
        
        if explorer_stats['temp_image_count'] + explorer_stats['pending_image_count']:        
            st.markdown("***")
            
            with fetch2:
                total_number_pending = explorer_stats['temp_image_count'] + explorer_stats['pending_image_count']
                if total_number_pending:
                    
                    if explorer_stats['temp_image_count'] == 0 and explorer_stats['pending_image_count'] > 0:
                        st.info(f"###### {explorer_stats['pending_image_count']} images pending generation")
                        button_text = "Check for new images"
                    elif explorer_stats['temp_image_count'] > 0 and explorer_stats['pending_image_count'] == 0:
                        st.info(f"###### {explorer_stats['temp_image_count']} new images generated")                        
                        button_text = "Pull new images"
                    else:
                        st.info(f"###### {explorer_stats['pending_image_count']} images pending generation and {explorer_stats['temp_image_count']} ready to be fetched")
                        button_text = "Check for/pull new images"
                    
                # st.info(f"###### {total_number_pending} images pending generation")
                # st.info(f"###### {explorer_stats['temp_image_count']} new images generated")     
                # st.info(f"###### {explorer_stats['pending_image_count']} images pending generation")     
            
            with fetch3:
                    if st.button(f"{button_text}", key=f"check_for_new_images_", use_container_width=True):
                        if explorer_stats['temp_image_count']:
                            data_repo.update_temp_gallery_images(project_uuid)
                            st.success("New images fetched")
                            time.sleep(0.3)
                        st.rerun()

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