from utils.data_repo.data_repo import DataRepo
import streamlit as st
from utils.state_refresh import refresh_app


def welcome_page():

    data_repo = DataRepo()
    app_setting = data_repo.get_app_setting_from_uuid()

    if app_setting.welcome_state == 0:
        st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")
        st.markdown(
            "#### To start, here are some weird, beautiful, and interesting things people have made with Dough and [Steerable Motion](https://github.com/banodoco/steerable-motion), the technology behind Dough"
        )

        st.markdown(
            "##### First, check out this insanity by the inimitable [Flipping Sigmas](https://www.youtube.com/watch?v=ETfiUYij5UE):"
        )
        vid1, vid2 = st.columns([1, 1])
        with vid1:
            st.video("https://banodoco.s3.amazonaws.com/plan/flipping_sigmas.mp4")

        st.markdown(
            "##### Next, look at what [Hannah Submarine](https://twitter.com/I_Han_naH_I) made for Grimes' Coachella set:"
        )
        vid1, vid2 = st.columns([1, 1])
        with vid1:
            st.video("https://banodoco.s3.amazonaws.com/plan/hannah_submarine.mp4")

        st.markdown(
            "##### Finally, here's a little video [I made](https://twitter.com/peteromallet) for a poem I love:"
        )

        vid1, vid2 = st.columns([1, 2])
        with vid1:
            st.video("https://banodoco.s3.amazonaws.com/plan/pom.mp4")

        st.markdown("***")

        footer1, footer2 = st.columns([2, 1])
        with footer1:
            st.markdown(
                "##### There's so much more you can do with Dough - we can't wait to see what you make!"
            )
            if st.button("I'm ready!", key="welcome_cta", type="primary", use_container_width=True):
                data_repo = DataRepo()
                data_repo.update_app_setting(welcome_state=1)
                refresh_app()

        st.markdown("***")

    elif app_setting.welcome_state == 1:
        welcome1, welcome2 = st.columns([1, 1])
        with welcome1:
            st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")
            st.subheader("Here are 6 things you should know about Dough before getting started:")

            st.markdown(
                """             
            
            1. The first shot you animate will probably look crap. Using Dough is a skill - you'll need to experiment, iterate and learn - but you'll learn quickly!     
            2. Dough won't work well for all styles or all types of motion - like all artistic mediums, it has its strengths and weaknesses. You'll need to figure out how to make it work for you.
            3. The app will be slow the first time you use each function. It needs to download models and install stuff. Please be patient (but check your Terminal for error messages if it's taking more than 15 minutes).
            4. This is a beta - **please** share any feedback you have - especially negative stuff. We're focused on making this better and we can't do that without your help. We won't be offended by anything you say! It also is buggier than we'd like right now - especially on Windows - please report any issues!
            5. You need to press 'r' for new generations and updates to appear - they won't show automatically.
            6. To make something great, you will need to use a video editor like DaVinci Resolve, Premiere Pro or even iMovie. Dough is the first step in the process but editing is what will make it magical.
            
            """
            )

            st.write("")

            read1, read2, _ = st.columns([1, 1, 1])
            with read1:

                actually_read = st.checkbox("I actually read that", key="actually_read")
            with read2:
                if actually_read:

                    if st.button("Continue to the app", key="welcome_cta"):
                        data_repo = DataRepo()
                        data_repo.update_app_setting(welcome_state=2)
                        refresh_app()
                else:
                    st.button(
                        "Continue to the app",
                        key="welcome_cta",
                        disabled=True,
                        help="You need to confirm on the left that you've read the note",
                    )
