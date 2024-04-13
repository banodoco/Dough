from utils.data_repo.data_repo import DataRepo
import streamlit as st

def welcome_page():
    
    
    data_repo = DataRepo()
    app_setting = data_repo.get_app_setting_from_uuid()
    if app_setting.welcome_state == 0:
        st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")          
        st.markdown('#### To start, here are some weird, beautiful, and interesting things people have made with Dough and [Steerable Motion](https://github.com/banodoco/steerable-motion), the technology behind Dough')

        vertical1, vertical2, vertical3 = st.columns([1,1,1])

        with vertical1:
            st.video("https://banodoco.s3.amazonaws.com/dough-website/vertical-3.mp4", format='mp4', start_time=0)
            st.link_button(url="https://www.youtube.com/watch?v=ETfiUYij5UE", label="By Flipping Sigmas", use_container_width=True)


        with vertical2:
            st.video("https://banodoco.s3.amazonaws.com/dough-website/vertical-2.mp4", format='mp4', start_time=0)
            st.link_button(url="https://www.instagram.com/midjourney.man/", label="By Midjourney Man", use_container_width=True)

        with vertical3:
            st.video("https://banodoco.s3.amazonaws.com/dough-website/vertical-1.mp4", format='mp4', start_time=0)
            st.link_button(url="https://www.instagram.com/superbeasts.ai", label="By Superbeasts", use_container_width=True)

        rectangle1, rectangle2  = st.columns([1,1])

        with rectangle1:
            st.video("https://banodoco.s3.amazonaws.com/dough-website/horizontal-1.mp4", format='mp4', start_time=0)
            st.link_button(url="https://twitter.com/I_Han_naH_I", label="By Hannah Submarine", use_container_width=True)

        with rectangle2:
            st.video("https://banodoco.s3.amazonaws.com/dough-website/horizontal-2.mp4", format='mp4', start_time=0)
            st.link_button(url="https://www.instagram.com/emma_catnip/?hl=en", label="By Emma Catnip", use_container_width=True)
            
        square1, square2, square3 = st.columns([1,1,1])


        with square1:
            st.video("https://banodoco.s3.amazonaws.com/dough-website/square-2.mp4", format='mp4', start_time=0)
            st.link_button(url="https://www.instagram.com/teslanaut/", label="By Teslanaut", use_container_width=True)

        with square2:
            st.video("https://banodoco.s3.amazonaws.com/dough-website/square-1.mov", format='mp4', start_time=0)
            st.link_button(url="https://twitter.com/I_Han_naH_I", label="By Hannah Submarine", use_container_width=True)

        with square3:
            st.video("https://banodoco.s3.amazonaws.com/dough-website/square_4.mp4", format='mp4', start_time=0)
            st.link_button(url="https://twitter.com/peteromallet", label="By POM", use_container_width=True)



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