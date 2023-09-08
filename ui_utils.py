import streamlit as st

def hide_sidebar():
    css ='''
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

def hide_footer():
    hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def hide_menu():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 