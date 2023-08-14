import time
from PIL import Image
import streamlit as st

from ui_components.common_methods import prompt_clip_interrogator
from utils.common_utils import save_or_host_file


def prompt_finder_element(project_uuid):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("What image would you like to find the prompt for?", type=[
                                     'png', 'jpg', 'jpeg'], key="prompt_file")
    which_model = st.radio("Which model would you like to get a prompt for?", ["Stable Diffusion 1.5", "Stable Diffusion 2"], key="which_model",
                           help="This is to know which model we should optimize the prompt for. 1.5 is usually best if you're in doubt", horizontal=True)
    best_or_fast = st.radio("Would you like to optimize for best quality or fastest speed?", [
                            "Best", "Fast"], key="best_or_fast", help="This is to know whether we should optimize for best quality or fastest speed. Best quality is usually best if you're in doubt", horizontal=True).lower()
    if st.button("Get prompts"):
        if not uploaded_file:
            return
        
        uploaded_file_path = f"videos/{project_uuid}/assets/resources/prompt_images/{uploaded_file.name}"
        img = Image.open(uploaded_file)
        hosted_url = save_or_host_file(img, uploaded_file_path)
        uploaded_file_path = hosted_url or uploaded_file_path

        prompt = prompt_clip_interrogator(uploaded_file_path, which_model, best_or_fast)
        
        # if not os.path.exists(f"videos/{project_uuid}/prompts.csv"):
        #     with open(f"videos/{project_uuid}/prompts.csv", "w") as f:
        #         f.write("prompt,example_image,which_model\n")
        # add the prompt to prompts.csv
        # with open(f"videos/{project_uuid}/prompts.csv", "a") as f:
        #     f.write(
        #         f'"{prompt}",videos/{project_uuid}/assets/resources/prompt_images/{uploaded_file.name},{which_model}\n')
        
        st.session_state["last_generated_prompt"] = prompt

        st.success("Prompt added successfully!")
        time.sleep(0.3)
        uploaded_file = ""
        st.experimental_rerun()
    
    if 'last_generated_prompt' in st.session_state and st.session_state['last_generated_prompt']:
        st.write("Generated prompt - ", st.session_state['last_generated_prompt'])
    
    # list all the prompts in prompts.csv
    # if os.path.exists(f"videos/{project_uuid}/prompts.csv"):

    #     df = pd.read_csv(f"videos/{project_uuid}/prompts.csv", na_filter=False)
    #     prompts = df.to_dict('records')

    #     prompts.reverse()

    #     col1, col2 = st.columns([1.5, 1])
    #     with col1:
    #         st.markdown("### Prompt")
    #     with col2:
    #         st.markdown("### Example Image")
    #     with open(f"videos/{project_uuid}/prompts.csv", "r") as f:
    #         for i in prompts:
    #             index_of_current_item = prompts.index(i)
    #             col1, col2 = st.columns([1.5, 1])
    #             with col1:
    #                 st.write(prompts[index_of_current_item]["prompt"])
    #             with col2:
    #                 st.image(prompts[index_of_current_item]
    #                          ["example_image"], use_column_width=True)
    #             st.markdown("***")
