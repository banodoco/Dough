from utils.data_repo.data_repo import DataRepo
import streamlit as st

def welcome_page():
    
    
    data_repo = DataRepo()
    app_setting = data_repo.get_app_setting_from_uuid()
    if app_setting.welcome_state == 0:
        st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")  
        st.subheader("To start, here are 3 simple examples of the kind of things you can make with Dough:")
        st.markdown("***")
        

        st.subheader("First, look at this guy go!")
        img1, img2, vid = st.columns([1.5,1.5,3])
        with img1:
            st.image("sample_assets/example_generations/guy-1.png")
        with img2:
            st.image("sample_assets/example_generations/guy-2.png")
        with vid:
            st.video("sample_assets/example_generations/guy-result.mp4", format='mp4', start_time=0)
        
        st.markdown("***")
        st.subheader("Next, check this transformation out!")
        img1, img2, vid = st.columns([1.5,1.5,3])
        
        with img1:
            
            # in teh sample_assets/example_generations folder
            st.image("sample_assets/example_generations/world-1.png")        
            st.image("sample_assets/example_generations/world-3.png")
        with img2:
            st.image("sample_assets/example_generations/world-2.png")
                                
            st.image("sample_assets/example_generations/world-4.png")
        with vid:
            st.video("sample_assets/example_generations/world-result.mp4", format='mp4', start_time=0)

        st.markdown("***")
        st.subheader("This is just the same lady over and over again but it still looks pretty cool!")
        img1, img2, vid = st.columns([1.5,1.5,3])
        with img1:
            st.image("sample_assets/example_generations/lady-1.png")
            st.image("sample_assets/example_generations/lady-1.png")
        with img2:
            st.image("sample_assets/example_generations/lady-1.png")
            st.image("sample_assets/example_generations/lady-1.png")
        with vid:
            st.video("sample_assets/example_generations/lady-result.mp4", format='mp4', start_time=0)

        st.markdown("***")
        footer1, footer2 = st.columns([1,1.5])
        with footer1:
            st.subheader("Once you're ready to move on, click the button below")
            if st.button("I'm ready!", key="welcome_cta", type="primary", use_container_width=True):
                data_repo = DataRepo()
                data_repo.update_app_setting(welcome_state=1)
                st.rerun()
    
    elif app_setting.welcome_state == 1:
        welcome1, welcome2 = st.columns([1,1])
        with welcome1:
            st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")  
            st.subheader("Here are 6 things you should know about Dough before getting started:")

            st.markdown('''             
            
            1. The first shot you animate will probably look crap. Using Dough is a skill - you'll need to experiment, iterate and learn - but you'll learn quickly.        
            2. Dough won't work well for all styles or all types of motion - like all artistic mediums, it has its strengths and weaknesses. You'll need to figure out how to make it work for you.
            3. The app will be slow the first time you use each function. It needs to download models and install stuff. Please be patient (but check your Terminal for error messages if it's taking more than 15 minutes).
            4. This is a beta - **please** share any feedback you have - especially negative stuff. We're focused on making this better and we can't do that without your help. We won't be offended by anything you say!
            5. You need to press 'r' for new generations and updates to appear - they won't appear automatically.
            6. To make something great, you will need to use a video editor like DaVinci Resolve, Premiere Pro or even iMovie. Dough is the first step in the process but editing is what makes it magical.
            
            ''')

            st.write("")

            read1, read2, _ = st.columns([1,1, 1])
            with read1:
                
                actually_read = st.checkbox("I actually read that", key="actually_read")
            with read2:
                if actually_read:
                    
                    if st.button("Continue to the app", key="welcome_cta"):
                        data_repo = DataRepo()
                        data_repo.update_app_setting(welcome_state=2)
                        st.rerun()
                else:
                    st.button("Continue to the app", key="welcome_cta", disabled=True, help="You need to confirm on the left that you've read the note")