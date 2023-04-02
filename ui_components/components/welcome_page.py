import streamlit as st

from repository.local_repo.csv_data import get_app_settings, update_app_settings

def welcome_page():
    app_settings = get_app_settings()
    
    def nav_buttons(step):
        button1, button2, button3 = st.columns([2,6,2])
        with button1:
            if step == 0:
                if st.button("Previous Step", disabled=True):
                    st.write("")
            else:
                if st.button("Previous Step"):
                    st.session_state["welcome_state"] = str(int(step) - 1)
                    update_app_settings("welcome_state", st.session_state["welcome_state"])                                    
                    st.experimental_rerun()
            
            if st.button("Skip Intro"):
                st.session_state["welcome_state"] = 7
                update_app_settings("welcome_state", st.session_state["welcome_state"]) 
                st.experimental_rerun()
        with button2:
            st.write("")
            
        with button3:
            if st.button("Next Step", type="primary"):
                st.session_state["welcome_state"] = step + 1  
                update_app_settings("welcome_state", st.session_state["welcome_state"])                  
                st.experimental_rerun()
     
    if int(st.session_state["welcome_state"]) == 0 and st.session_state["online"] == False:
        st.header("Welcome to Banodoco!")                
        st.subheader("First, a quick demo!")            
        st.write("I've put together a quick demo video to show you how to use the app. While I recommend you watch it, you can also click the button to skip it and go straight to the app.")
        st.video("https://youtu.be/YQkwcsPGLnA")
        nav_buttons(int(st.session_state["welcome_state"]))

    elif int(st.session_state["welcome_state"]) == 1 and st.session_state["online"] == False:
        st.subheader("Next, a example of a video made with it!")
        st.write("I've put together a quick video to show you how to use the app. While I recommend you watch it, you can also click the button to skip it and go straight to the app.")
        st.video("https://www.youtube.com/watch?v=vWWBiDjwKkg&t")
        nav_buttons(int(st.session_state["welcome_state"]))

    elif int(st.session_state["welcome_state"]) == 2 and st.session_state["online"] == False:
        st.subheader("And here's a more abstract video made with it...")
        st.write("I've put together a quick video to show you how to use the app. While I recommend you watch it, you can also click the button to skip it and go straight to the app.")
        st.video("https://youtu.be/ynJyxnEzepM")
        nav_buttons(int(st.session_state["welcome_state"]))

    elif int(st.session_state["welcome_state"]) == 3 and st.session_state["online"] == False:
        st.subheader("Add your Replicate credentials")
        st.write("Currently, we use Replicate.com for our model hosting. If you don't have an account, you can sign up for free [here](https://replicate.com/signin) and grab your API key [here](https://replicate.com/account) - this data is stored locally on your computer.")
        with st.expander("Why Replicate.com? Can I run the models locally?"):
            st.info("Replicate.com allows us to rapidly implement a wide array of models that work on any computer. Currently, these are delivered via API, which means that you pay for GPU time - but it tends to be very cheap. Getting it locally [via COG](https://github.com/replicate/cog/blob/main/docs/wsl2/wsl2.md) shouldn't be too difficult but I don't have the hardware to do that")
        st.session_state["replicate_user_name"] = st.text_input("replicate_user_name", value = app_settings["replicate_user_name"])
        st.session_state["replicate_com_api_key"]  = st.text_input("replicate_com_api_key", value = app_settings["replicate_com_api_key"])
        st.warning("You can add this in App Settings later if you wish.")
        nav_buttons(int(st.session_state["welcome_state"]))

    elif int(st.session_state["welcome_state"]) == 4 and st.session_state["online"] == False:
        st.subheader("That's it! Just click below when you feel sufficiently welcomed, and you'll be taken to the app!")                        
        if st.button("I feel welcomed!", type="primary"):
            st.balloons()
            if st.session_state["replicate_com_api_key"] != "":
                update_app_settings("replicate_user_name", st.session_state["replicate_user_name"])
            if st.session_state["replicate_com_api_key"] != "":
                update_app_settings("replicate_com_api_key", st.session_state["replicate_com_api_key"])
            update_app_settings("welcome_state", 7)
            st.session_state["welcome_state"] = 7
            st.experimental_rerun()