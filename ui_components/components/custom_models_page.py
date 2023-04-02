import ast
import os
import time
import pandas as pd
import streamlit as st

from ui_components.common_methods import get_model_details, get_models, train_model
from repository.local_repo.csv_data import get_app_settings


def custom_models_page(project_name):
    app_settings = get_app_settings()
    with st.expander("Existing models"):
        
        st.subheader("Existing Models:")

        models = get_models() 
        if models == []:
            st.info("You don't have any models yet. Train a new model below.")
        else:
            header1, header2, header3, header4, header5, header6 = st.columns(6)
            with header1:
                st.markdown("###### Model Name")
            with header2:
                st.markdown("###### Trigger Word")
            with header3:
                st.markdown("###### Model ID")
            with header4:
                st.markdown("###### Example Image #1")
            with header5:
                st.markdown("###### Example Image #2")
            with header6:                
                st.markdown("###### Example Image #3")
            
            for i in models:   
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    model_details = get_model_details(i)
                    model_details["name"]
                with col2:
                    if model_details["keyword"] != "":
                        model_details["keyword"]
                with col3:
                    if model_details["keyword"] != "":
                        model_details["id"]
                with col4:
                    st.image(ast.literal_eval(model_details["training_images"])[0])
                with col5:
                    st.image(ast.literal_eval(model_details["training_images"])[1])
                with col6:
                    st.image(ast.literal_eval(model_details["training_images"])[2])
                st.markdown("***")            

    with st.expander("Train a new model"):
        st.subheader("Train a new model:")
        
        type_of_model = st.selectbox("Type of model:",["LoRA","Dreambooth"],help="If you'd like to use other methods for model training, let us know - or implement it yourself :)")
        model_name = st.text_input("Model name:",value="", help="No spaces or special characters please")
        if type_of_model == "Dreambooth":
            instance_prompt = st.text_input("Trigger word:",value="", help = "This is the word that will trigger the model")        
            class_prompt = st.text_input("Describe what your prompts depict generally:",value="", help="This will help guide the model to learn what you want it to do")
            max_train_steps = st.number_input("Max training steps:",value=2000, help=" The number of training steps to run. Fewer steps make it run faster but typically make it worse quality, and vice versa.")
            type_of_task = ""
            resolution = ""
            
        elif type_of_model == "LoRA":
            type_of_task = st.selectbox("Type of task:",["Face","Object","Style"]).lower()
            resolution = st.selectbox("Resolution:",["512","768","1024"],help="The resolution for input images. All the images in the train/validation dataset will be resized to this resolution.")
            instance_prompt = ""
            class_prompt = ""
            max_train_steps = ""
        uploaded_files = st.file_uploader("Images you'd like to train the model based on:", type=['png','jpg','jpeg'], key="prompt_file",accept_multiple_files=True)
        if uploaded_files is not None:   
            column = 0                             
            for image in uploaded_files:
                # if it's an even number 
                if uploaded_files.index(image) % 2 == 0:
                    column = column + 1                                   
                    row_1_key = str(column) + 'a'
                    row_2_key = str(column) + 'b'                        
                    row_1_key, row_2_key = st.columns([1,1])
                    with row_1_key:
                        st.image(uploaded_files[uploaded_files.index(image)], width=300)
                else:
                    with row_2_key:
                        st.image(uploaded_files[uploaded_files.index(image)], width=300)
                                                                                
            st.write(f"You've selected {len(uploaded_files)} images.")
            
        if len(uploaded_files) <= 5 and model_name == "":
            st.write("Select at least 5 images and fill in all the fields to train a new model.")
            st.button("Train Model",disabled=True)
        else:
            if st.button("Train Model",disabled=False):
                st.info("Loading...")
                images_for_model = []
                for image in uploaded_files:
                    with open(os.path.join(f"training_data",image.name),"wb") as f: 
                        f.write(image.getbuffer())                                                        
                        images_for_model.append(image.name)                                                  
                model_status = train_model(app_settings,images_for_model, instance_prompt,class_prompt,max_train_steps,model_name, project_name, type_of_model, type_of_task, resolution)
                st.success(model_status)

    with st.expander("Add model from internet"):
        st.subheader("Add a model the internet:")    
        uploaded_type_of_model = st.selectbox("Type of model:",["LoRA","Dreambooth"], key="uploaded_type_of_model", disabled=True, help="You can currently only upload LoRA models - this will change soon.")
        uploaded_model_name = st.text_input("Model name:",value="", help="No spaces or special characters please", key="uploaded_model_name")                                   
        uploaded_model_images = st.file_uploader("Please add at least 2 sample images from this model:", type=['png','jpg','jpeg'], key="uploaded_prompt_file",accept_multiple_files=True)
        uploaded_link_to_model = st.text_input("Link to model:",value="", key="uploaded_link_to_model")
        st.info("The model should be a direct link to a .safetensors files. You can find models on websites like: https://civitai.com/" )
        if uploaded_model_name == "" or uploaded_link_to_model == "" or uploaded_model_images is None:
            st.write("Fill in all the fields to add a model from the internet.")
            st.button("Upload Model",disabled=True)
        else:
            if st.button("Upload Model",disabled=False):                    
                images_for_model = []
                for image in uploaded_model_images:
                    with open(os.path.join(f"training_data",image.name),"wb") as f: 
                        f.write(image.getbuffer())                                                        
                        images_for_model.append(image.name)
                for i in range(len(images_for_model)):
                    images_for_model[i] = 'training_data/' + images_for_model[i]
                df = pd.read_csv("models.csv")
                df = df.append({}, ignore_index=True)
                new_row_index = df.index[-1]
                df.iloc[new_row_index, 0] = uploaded_model_name
                df.iloc[new_row_index, 4] = str(images_for_model)
                df.iloc[new_row_index, 5] = uploaded_type_of_model
                df.iloc[new_row_index, 6] = uploaded_link_to_model
                df.to_csv("models.csv", index=False)
                st.success(f"Successfully uploaded - the model '{model_name}' is now available for use!")
                time.sleep(1.5)
                st.experimental_rerun()