import time
import streamlit as st
import webbrowser
from shared.constants import SERVER, ServerType
from utils.common_utils import get_current_user

from utils.data_repo.data_repo import DataRepo


def app_settings_page():
    data_repo = DataRepo()
            
    if SERVER == ServerType.DEVELOPMENT.value:
        st.subheader("Purchase Credits")
        st.write("This feature is only available in production")

    if SERVER != ServerType.DEVELOPMENT.value:
        with st.expander("Purchase Credits", expanded=True):
            user_credits = get_current_user().total_credits
            user_credits = round(user_credits, 2) if user_credits else 0
            st.write(f"Total Credits: {user_credits}")
            c1, c2 = st.columns([1,1])
            with c1:
                if 'input_credits' not in st.session_state:
                    st.session_state['input_credits'] = 10

                credits = st.number_input("Credits (1 credit = $1)", value = st.session_state['input_credits'], step = 10)
                if credits != st.session_state['input_credits']:
                    st.session_state['input_credits'] = credits
                    st.rerun()

                if st.button("Generate payment link"):
                    if credits < 10:
                        st.error("Minimum credit value should be atleast 10")
                        time.sleep(0.7)
                        st.rerun()
                    else:
                        payment_link = data_repo.generate_payment_link(credits)
                        payment_link = f"""<a target='_self' href='{payment_link}'> PAYMENT LINK </a>"""
                        st.markdown(payment_link, unsafe_allow_html=True)