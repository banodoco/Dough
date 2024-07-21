from utils.data_repo.data_repo import DataRepo
import streamlit as st
from utils.state_refresh import refresh_app
import time


def welcome_page():

    data_repo = DataRepo()
    app_setting = data_repo.get_app_setting_from_uuid()

    if app_setting.welcome_state == -1:
        st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")

        vid1, vid2 = st.columns([2, 1])
        with vid1:
            st.markdown(
                "#### To start, here are some weird, beautiful, and interesting things people have made with Dough and [Steerable Motion](https://github.com/banodoco/steerable-motion), the technology behind Dough"
            )

            vertical1, vertical2, vertical3 = st.columns([1, 1, 1])

            with vertical1:
                st.video(
                    "https://banodoco.s3.amazonaws.com/plan/fabdream_resized_cropped.mp4",
                    format="mp4",
                    start_time=0,
                )
                st.link_button(
                    url="https://www.instagram.com/fabdream.ai/?hl=en",
                    label="By Fabdream",
                    use_container_width=True,
                )

            with vertical2:
                st.video(
                    "https://banodoco.s3.amazonaws.com/dough-website/vertical-2.mp4",
                    format="mp4",
                    start_time=0,
                )
                st.link_button(
                    url="https://www.instagram.com/midjourney.man/",
                    label="By Midjourney Man",
                    use_container_width=True,
                )

            with vertical3:
                st.video("https://banodoco.s3.amazonaws.com/plan/chris_exe.mov", format="mp4", start_time=0)
                st.link_button(
                    url="https://www.instagram.com/syntaxdiffusion/",
                    label="By syntaxdiffusion",
                    use_container_width=True,
                )

            rectangle1, rectangle2 = st.columns([1, 1])
            with rectangle1:
                st.video(
                    "https://banodoco.s3.amazonaws.com/dough-website/horizontal-1.mp4",
                    format="mp4",
                    start_time=0,
                )
                st.link_button(
                    url="https://twitter.com/I_Han_naH_I",
                    label="By Hannah Submarine",
                    use_container_width=True,
                )

            with rectangle2:
                st.video("https://banodoco.s3.amazonaws.com/plan/byarloo.mp4", format="mp4", start_time=0)
                st.link_button(
                    url="https://www.instagram.com/byarlooo/", label="By ARLO", use_container_width=True
                )

            square1, square2, square3 = st.columns([1, 1, 1])

            with square1:
                st.video(
                    "https://banodoco.s3.amazonaws.com/dough-website/square-1.mov", format="mp4", start_time=0
                )
                st.link_button(
                    url="https://twitter.com/I_Han_naH_I",
                    label="By Hannah Submarine",
                    use_container_width=True,
                )

            with square2:
                st.video(
                    "https://banodoco.s3.amazonaws.com/dough-website/square-2.mp4", format="mp4", start_time=0
                )
                st.link_button(
                    url="https://www.instagram.com/teslanaut/", label="By Teslanaut", use_container_width=True
                )

            with square3:
                # st.video("https://banodoco.s3.amazonaws.com/dough-website/square-3.mp4", format='mp4', start_time=0)
                st.video("https://banodoco.s3.amazonaws.com/pom.mp4", format="mp4", start_time=0)

                st.link_button(
                    url="https://twitter.com/peteromallet", label="By POM", use_container_width=True
                )

            st.markdown("***")
            st.markdown(
                "##### There's so much more you can do with Dough - we can't wait to see what you make!"
            )
            if st.button("I'm inspired!", key="welcome_cta", type="primary", use_container_width=True):
                data_repo = DataRepo()
                data_repo.update_app_setting(welcome_state=0)
                refresh_app()

        st.markdown("***")

    elif app_setting.welcome_state == 0:
        welcome1, welcome2 = st.columns([1.5, 2])
        with welcome1:
            st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")
            st.subheader("Here are 5 things you should know about Dough before getting started:")

            st.markdown(
                """             
            1. You need to **press 'r' for new generations to appear** - they won't show automatically.
            2. The first shot you animate will probably look crap. **Using Dough is a skill** - you'll need to experiment, iterate and learn to make good stuff!
            3. Dough **won't work well for all styles** or all types of motion - like all artistic mediums, it has its strengths and weaknesses. You'll need to figure out how to make it work for you.
            4. The app will be **slow the first time you use each function**. It needs to download models and install stuff. Please be patient (but check your Terminal for error messages if it's taking more than 15 minutes).                        
            5. To make something great, **you'll need to put in a lot of effort**, including using a video editor. Dough is the first step in the process but editing is what will make it magical.
            
            """
            )

            st.write("")

            read1, read2, _ = st.columns([1, 1, 1])
            with read1:

                actually_read = st.checkbox("I actually read that", key="actually_read")
            with read2:
                if actually_read:

                    if st.button("Continue", key="welcome_cta"):
                        data_repo = DataRepo()
                        data_repo.update_app_setting(welcome_state=1)
                        refresh_app()
                else:
                    st.button(
                        "Continue",
                        key="welcome_cta",
                        disabled=True,
                        help="You need to confirm on the left that you've read the note",
                    )
    elif app_setting.welcome_state == 1:
        welcome1, _ = st.columns([1, 1])
        with welcome1:
            st.subheader("Pop-quiz time!")
            st.info("##### Which of these is an unusual quirk of Dough's user experience?")
            st.caption("Once you've answered correctly, you'll be redirected to the app:")
            if st.button("It secretly mines cryptocurrency in the background", key="quiz1"):
                st.error("No, but that's on the roadmap for 2.0.")
                time.sleep(1)
                refresh_app()
            if st.button(
                "It's part of an elaborate plot to take help open source AI art thrive", key="quiz4"
            ):
                st.error("Yeah, but that's not a quirk of the user experience.")
                time.sleep(1)
                refresh_app()
            if st.button("You need to press 'r' for new generations to appear", key="quiz3"):
                st.success("Correct! You're ready to start using Dough.")
                time.sleep(2)
                data_repo = DataRepo()
                data_repo.update_app_setting(welcome_state=2)
                refresh_app()
            if st.button(
                "It randomly alters your system clock so you miss meetings",
                key="quiz2",
            ):
                st.error("We wll implement that if users time in app is too low.")
                time.sleep(1)
                refresh_app()
