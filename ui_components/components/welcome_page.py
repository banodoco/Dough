from utils.data_repo.data_repo import DataRepo
import streamlit as st

def welcome_page():
    welcome1, welcome2 = st.columns([1,1])
    with welcome1:
        st.subheader("Welcome! Here are 8 things to keep in mind as you use Dough:")

        st.markdown('''                
        1. The first shot you animate will probably look like crap. You'll need to try things, iterate and learn.
        2. The 4th shot you may will probably look pretty good. You will learn!
        3. Over time, you'll build of knowledge of how to compose shots: you'll learn what images work best and how to guide the motion precisely. Over time, this will allow you to create shots that are stunningly beautiful and uniquely you
        4. If you combine a bunch of these shots together, you can make scenes. Combine a bunch of these scenes and you could have a movie!
        5. Dough is a tool with a lot of opinions - it wants help you to create narrative-driven, shot-based animations. It wants you to craft each frame and shot to perfection. You don't have to listen to it but maybe you should!
        6. Dough won't work well for all styles: some tend to bring out the flaws in SD 1.5-based video models and the motion - e.g. realism tends to look slightly weird so I would avoid it. Realism is boring anyway!
        7. Our ultimate goal is to build a tool-builder that makes it easy for anyone to build artistic tools. While you're using this, maybe think of what tool you would create if it were ridiculously easy to do so.
        8. This is a beta - **please** share any feedback you have - especially negative feedback. If something doesn't work please let us know and we'll fix it. If some change could make your life easier, let us know and we'll implement it. 
        9. The app will be slow the first time you use each function. It needs to download models and install stuff!
        ''')
        
        if st.button("I feel welcomed", key="welcome_cta"):
            data_repo = DataRepo()
            data_repo.update_app_setting(welcome_state=1)
            st.rerun()