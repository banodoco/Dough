from utils.data_repo.data_repo import DataRepo
import streamlit as st

def welcome_page():
    welcome1, welcome2 = st.columns([1,1])
    with welcome1:
        st.markdown("# :green[D]:red[o]:blue[u]:orange[g]:green[h] :red[□] :blue[□] :orange[□]")  
        st.subheader("This is the last time you'll see this note")

        st.markdown('''             

        Here are 5 things you *actually* should know about Dough before getting started:
        1. The first shot you animate will probably look crap. Using Dough is a skill - you'll need to experiment, iterate and learn - but you'll learn quickly.        
        2. Dough won't work well for all styles - for example, realism tends to bring out the flaws in SD1.5-based models. (note: realism is boring anyway)
        3. The app will be slow the first time you use each function. It needs to download models and install stuff. Please be patient - but check your Terminal for error messages if it's taking more than 15 minutes.
        4. This is a beta - **please** share any feedback you have - especially negative stuff. We're relentlessly focused on making this better and we can't do that without your help. We won't be offended by anything you say!
        5. Our ultimate goal is to create a tool-builder that makes it easy for anyone to build artistic tools. While you're using this, think of what tool you would create if it were ridiculously easy to do so.
        
        ''')

        st.write("")

        read1, read2, _ = st.columns([1,1, 1])
        with read1:
            
            actually_read = st.checkbox("I actually read that", key="actually_read")
        with read2:
            if actually_read:
                
                if st.button("Continue to the app", key="welcome_cta"):
                    data_repo = DataRepo()
                    data_repo.update_app_setting(welcome_state=1)
                    st.rerun()
            else:
                st.button("Continue to the app", key="welcome_cta", disabled=True, help="You need to confirm on the left that you've read the note")