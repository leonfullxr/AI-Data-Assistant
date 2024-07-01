# import required libraries
import os
from apikey import api_key

import streamlit as st
import pandas as pd

from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

# OpenAI API key
os.environ['OPENAI_API_KEY'] = api_key
load_dotenv(find_dotenv())

# Large language model
llm = OpenAI(temperature=0) # Set the temperature to 0 to get deterministic results, no randomness

# Main

st.title('AI Assistant for Data Science')
st.header('Welcome to the Ai Assistant for Data Science!')
st.subheader('Solution')
st.write('Hello, I am your AI Assistant and I am here to help you with your Data Science Projects.')


with st.sidebar:
    st.write('*Your Data Science Adventure begins with an CSV File.*')
    st.caption('**Upload your CSV File and I will help you with the rest.**')
    
    
    
    st.divider()
    
    st.caption("<p style ='text-align:center'> Made by leonfullxr. </p>", unsafe_allow_html=True)
    
# Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}
    
# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True
st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    st.header('Exploratory Data Analysis Part')
    st.subheader('Solution')
    
    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)
        
with st.sidebar:
    with st.expander('What are the steps of EDA'):
        st.write(llm('What are the steps of EDA'))
        
pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)
question = 'What is the meaning of the columns'
columns_meaning = pandas_agent.run(question)
st.write(columns_meaning)