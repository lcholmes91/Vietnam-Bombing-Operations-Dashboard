# Luke Holmes
# Intro page to Vietnam Bombing Dashboard
# This is what's known as the "Entry Point" page for the Streamlit app. It provides an 
# overview of the dashboard, its purpose, and the data source. It also sets up the page
# configuration and title. The scripts in the "pages" folder contain the subsequent pages
# of the dashboard, which are accessible via the sidebar navigation in Streamlit.


import os
import polars as pl
import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI
import pydeck as pdk
# Must 'pip install sqlalchemy' & 'pip install "polars[sqlalchemy]" '
# Must 'pip install connectorx'
# Must 'pip install tabulate'

# --------------------------------------
# WEBPAGE CONFIGURATION
# --------------------------------------

st.set_page_config(                                     # Streamlit page configuration
    page_title="Vietnam War Aerial Bombing Dashboard",  # Page title
    layout="wide",                                      # Use wide layout
    page_icon="✈️"
)

# --------------------------------------
# TITLE & SUBTITLE
# --------------------------------------

st.markdown(                            # Centered dashboard title
    "<h1 style='text-align: center;'>Vietnam War Aerial Bombing Dashboard</h1>",
    unsafe_allow_html=True              # Render HTML safely - not plain text
)

st.markdown(                            # Centered dashboard subtitle
    "<h3 style='text-align: center;'>An interactive dashboard exploring the aerial bombing campaigns during the Vietnam War (1964-1973)</h3>",
    unsafe_allow_html=True              # Render HTML safely - not plain text
)

st.markdown("---")                    # Horizontal rule separator


# --------------------------------------
# DESCRIPTION OF DASHBOARD PAGES
# --------------------------------------

st.markdown(
    """
    ### Dashboard Overview

    This dashboard provides an interactive exploration of the aerial bombing campaigns during the Vietnam War (1964-1973):

    **Descriptive Statistics**: offers a comprehensive overview of the bombing data, including key statistics, trends over time, geographical distributions of targets, and the military services involved

    Navigate through the pages using the sidebar to explore different facets of the Vietnam War aerial bombing data. Feel free to interact with the visualizations and ask the dashboard's AI assistant at the bottom of each page to gain insights into the historical context of the conflict.

    ### Data Source
    The data used in this dashboard is sourced from the Theater History of Operations Reports (THOR) [Vietnam War Bombing Operations dataset](https://www.kaggle.com/datasets/usaf/vietnam-war-bombing-operations?select=THOR_Vietnam_Bombing_Operations.csv), which provides:
    - A bombing operation .csv file which contains each mission’s 5Ws
    - An aircraft glossary explaining the models of U.S. aircraft used during Vietnam War
    - A glossary of munitions used by the aircraft
    - An explanatory dictionary for THOR data
    """,
    unsafe_allow_html=True              # Render HTML safely - not plain text
)

# --------------------------------------
# LLM HELPER FUNCTIONS
# --------------------------------------

# After the user inputs the filter parameters (i.e. start/end year, aircraft, etc.), 
# under the hood the dashboard has a Polars df with the returned query dat. We can give 
# the LLM summary stats or small samples of this df to help it answer questions. We can 
# feed some of this df (i.e. small samples of rows to show patterns, summary stats, etc.)
# plus the user's questions to the LLM as context to the model to produce better answers.
def is_llm_configured():
    """Check whether Streamlit sees your NVIDIA API key"""    
    return "NVIDIA_API_KEY" in st.secrets

def run_llm_intro(user_msg: str, history: list):
    """Call NVIDIA's hosted model we set in secrets.toml
       with some context from the currently filtered dataframe.
       This function is for the INTRO PAGE - no df is passed in.
    --------------------------------------
    Args:
        user_msg (str): user's inputs into the chat
        history (list[dicts[str]]): record of the user's chat messages for context
    --------------------------------------
    Returns:
        string: LLM's response to the user's inputs (comes from a ChatCompletion object)
    """    
    client = OpenAI(
        base_url = st.secrets["NVIDIA_API_BASE"],   # NVIDIA API base URL
        api_key = st.secrets["NVIDIA_API_KEY"]      # NVIDIA API key
    )

    # --------------------------------------
    # LLM SYSTEM PROMPT
    # --------------------------------------
    system_prompt = f"""
    You are an assistant helping a user explore a dataset of Vietnam War bombing missions. 
    The user is interacting with a multi-page Streamlit dashboard that filters the data by year and
    aircraft type. The user is currently on the INTRO PAGE of the dashboard, and they may ask
    questions about the Vietnam War, aerial bombing campaigns, or the dataset itself. The dataset was
    sourced from the Theater History of Operations Reports (THOR) Vietnam War Bombing Operations dataset on 
    Kaggle (https://www.kaggle.com/datasets/usaf/vietnam-war-bombing-operations?select=THOR_Vietnam_Bombing_Operations.csv).

    When you answer:
    - Speak **directly to the user** in a natural, conversational tone.
    - Use the second persion ("you") and past tense where appropriate.
    - Do **not** describe what you are about to do.
    - Do **not** say things like "we need to respond to the user" or "the assistant should".
    - Just give the final answer, as if you were chatting with the user.

    Use general knowledge of the Vietnam War to answer the user's questions about the general context of the Vietnam
    War or the dataset. If you don't have enough information to be precise, say so explicitly. Do NOT make up exact 
    numeric values that aren't implied by the summary above. If the user asks for specific statistics or data points 
    from the dataset, politely inform them that you don't have access to the data on this page, and 
    suggest they navigate to the Descriptive Statistics page before providing any other contextual information. 
    """

    # Build messages list (system + history + new user messages)
    # Ea. message to/from the LLM is a dict with "role" & "content" keys
    messages = [{"role": "system",                  # Add system prompt first
                 "content": system_prompt}]
    messages.extend(history)                        # Add prior chat history (mult. messages)
    messages.append({"role": "user",                # Add the new user message last
                     "content": user_msg})

    # This is the critical call to the NVIDIA LLM API
    # The response will be a ChatCompletion object, which contains the model's 
    # reply to the user's message inside response.choices[0].message.content
    # Some of these parameters (max_tokens, top_p, temperature) can be tuned
    # and affect the randomness of text generation. Temperature [0,1] reshapes the 
    # probability distribution of all possible next words. Top-p [0,1] creates a dynamic 
    # cutoff by selecting from the most likely words that cumulatively reach a 
    # certain probability threshold.
    response = client.chat.completions.create(
        model = st.secrets["NVIDIA_MODEL"],     # Model name from secrets.toml
        max_tokens=4096,                        # Max tokens in the response
        top_p=1,                                # Lower top_p = smaller set of most likely words, higher top_p = larger pool of words considered
        temperature=0.3,                        # Low temp = less random, high temp = more random
        messages=messages,                      # The messages we built above
    )

    return response.choices[0].message.content  # Return the LLM's reply text


# --------------------------------------
# CHAT ASSISTANT UI
# --------------------------------------
st.markdown("---")                          # Horizontal grey line to separate sections
st.subheader("Ask the dashboard assistant...")      # Chat assistant subheader

# Fail gracefully if the LLM isn't configured
if not is_llm_configured():                 # If the LLM isn't configured
    st.info(                                # Show info message - point the user to the README
        "LLM assistant not configured. "
        "If you're running this app yourself, add your NVIDIA API key to "
        "`.streamlit/secrets.toml` (see README)"
    )
else:                                       # If the LLM is configured
    if "messages" not in st.session_state:  # Initialize chat history in session state
        st.session_state.messages = []      # Empty list to hold messages - each message is a dict with "role" & "content" keys

    for msg in st.session_state.messages:   # Display all prior messages in the chat history
        with st.chat_message(msg["role"]):  # Role is either "user" or "assistant"
            st.markdown(msg["content"])     # Content is the message text - display with markdown formatting

    # Chat input
    user_msg = st.chat_input(               # Input box for user to type messages
        "Ask me a question about the Vietnam War, this dashboard's dataset, etc."
    )

    if user_msg:                                    # If the user submitted a message
        st.session_state.messages.append(           # Save user message to history
            {"role": "user", "content": user_msg}   # Remember, it's the 'user' role
        )                                           

        with st.chat_message("user"):               # Display the user's message in the chat
            st.markdown(user_msg)

        # Here's the critical part - call the LLM API with the user's message, 
        # the filtered df, & the prior chat history for context
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):                     # Show a "thinking" spinner while waiting for LLM response
                try:
                    assistant_reply = run_llm_intro(            # Call our LLM helper function
                        user_msg=user_msg,                      # User's message
                        history=st.session_state.messages[:-1]  # Prior chat history - exclude current user message...
                    )                                           # so it doesn't appear twice
                
                except Exception as e:                          # If there's an error calling the LLM API
                    assistant_reply = f"Sorry, I couldn't reach the LLM API: `{e}`"

                st.markdown(assistant_reply)                    # Display the LLM's reply in the chat

        # Save assistant reply to history
        st.session_state.messages.append(
            {"role": "assistant", 
             "content": assistant_reply}
        )