# Luke Holmes
# This script is what streamlit will read to build a dashboard of our visualizations.
# I used ChatGPT-5.1 to learn about how to build dashboards with Streamlit, Polars, etc.,
# and how to connect it to an SQLite database. I also used the NVIDIA LLM API docs to learn how to
# connect to their hosted models. I'll narrate the code with comments to explain each section.

# All these dependencies need to be in the requirements.txt file for deployment
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
# DATABASE FUNCTIONS
# --------------------------------------

# Create a database locally if one doesn't already exist - the dashboard will need one to reference
# locally (if running locally) or in the cloud (if running in the cloud) where it will host the dashboard
#DB_PATH = "Data/vietnam.db"                                     # Path to the SQLite DB file
PARQUET_PATH = "Data/THOR_Vietnam_Bombing_Operations.parquet"   # Path to the source .parquet file

@st.cache_data      # Caches function outputs that depend on inputs (i.e. results of SQL queries)
def load_data(yr_start: int, yr_end: int) -> pl.DataFrame:
    """Load data from Parquet file and filter by year range.
    --------------------------------------
    Args:
        yr_start (int): Start year for filtering
        yr_end (int): End year for filtering
    --------------------------------------
    Returns:
        pl.DataFrame: Filtered Polars DataFrame
    """    
    df = pl.read_parquet(PARQUET_PATH)                              # Read in the .parquet w/ Polars

    # Convert MSNDATE to Datetime
    df = df.with_columns(pl.col("MSNDATE").str.strptime(pl.Datetime, strict=False))

    df = (df.filter(
        pl.col("MSNDATE").dt.year().is_between(yr_start, yr_end)).select(
            ["MSNDATE", 
             "TGTLATDD_DDD_WGS84", 
             "TGTLONDDD_DDD_WGS84", 
             "NUMWEAPONSDELIVERED", 
             "VALID_AIRCRAFT_ROOT", 
             "MFUNC_DESC", 
             "MILSERVICE"]))
    return df



# Every time we run a query, we want to make sure the DB exists first.
# If it doesn't, we create it from the .parquet file. The DB is created
# on the user's disc, and because we've listed all .db files in .gitignore,
# it won't be committed to the GitLab repo.
#def ensure_db():
#    """Creates an SQLite DB from .parquet file if it doesn't exist yet."""       
    
#    if os.path.exists(DB_PATH):                                     # If the DB already exists
#        return                                                      # Do nothing
    
#    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)            # Make sure the data directory exists

#    print("Building SQLite database from Parquet (first run only)...")
#    df = pl.read_parquet(PARQUET_PATH)                              # Read in the .parquet w/ Polars

    # Build SQLite connection Uniform Resource Identifier (URI); in DB operations,
    # this is a string that describes how to connect to a DB. Libraries like SQLAlchemy 
    # and Polars understand this string and use it to open a connection. Polars helpers
    # are built around URIs (or engines), not around raw sqlite3 connection objects.
    # sqlite = which DB driver/type to use
    # :///   = part of the URI syntax, meaning "use a local file"
    # os.path... = the DB file path
#    db_uri = "sqlite:///" + os.path.abspath(DB_PATH)    # Absolute path to vietnam.db

    # Write Polars DataFrame to SQLite in one go
#    df.write_database(              # Polars version of to_sql()
#        table_name = "missions",    # Create a new table
#        connection = db_uri,        # Open the connection; since we're using URI, we don't manually connect
#        if_table_exists="replace",  # Replace existing tables
#        engine = "sqlalchemy"       # Default 
#    )
#    return


# @st.cache_... is what's known as a Streamlit "caching decoration", which is used to avoid 
# doing expensive work over & over when the app re-runs.
#@st.cache_resource  # Caches long-lived objects that you want to create once and reuse (i.e. DB connections)
#def get_db_uri():
#    """Creates an SQLite DB (on first run if needed) & establishes 
#       a connection to that DB
#    --------------------------------------
#    Returns:
#        URI string to the Vietnam DB
#    """
#    ensure_db()                                     # Calls our DB helper function
#    return "sqlite:///" + os.path.abspath(DB_PATH)  # Return the URI string = "this is an SQLite DB & its file path"

#@st.cache_data      # Caches function outputs that depend on inputs (i.e. results of SQL queries)
#def run_query(sql: str):
#    uri = get_db_uri()                              # Get the DB URI
#    return pl.read_database_uri(                    # Return a Polars DF from the DB using...,
#        query=sql,                                  # ...the SQL query from our sidebar filters, 
#        uri=uri,                                    # using the DB URI we just got
#    )                                               # Note this is using a SQLAlchemy engine under the hood

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

def run_llm(user_msg: str, df: pl.DataFrame, history: list):
    """Call NVIDIA's hosted model we set in secrets.toml
       with some context from the currently filtered dataframe.
    --------------------------------------
    Args:
        user_msg (str): user's inputs into the chat
        df (pl.DataFrame): user's currently set filters from the sidebar
        history (list[dicts[str]]): record of the user's chat messages for context
    --------------------------------------
    Returns:
        string: LLM's response to the user's inputs (comes from a ChatCompletion object)
    """    
    client = OpenAI(
        base_url = st.secrets["NVIDIA_API_BASE"],   # NVIDIA API base URL
        api_key = st.secrets["NVIDIA_API_KEY"]      # NVIDIA API key
    )

    # Handle if empty gracefully
    if df.height == 0:                              # If there are no rows in the filtered df    
        summary_text = "The currently filtered dataset has 0 missions." 
        sample_text = "(no sample rows because there are no matching missions)"
    else:                                           # If there are rows in the filtered df
        # Summary statistics
        summary = df.select(                        # Compute summary stats
            [
                pl.len().alias("num_missions"),     # Number of missions
                pl.col("NUMWEAPONSDELIVERED")       # Total weapons delivered
                    .cast(pl.Int64)
                    .sum()
                    .alias("total_weapons")
            ]
        ).to_dicts()[0]
        # Summary text   
        summary_text = (                            # Formatted summary text (will go in system prompt)
            f"- Number of missions: {summary['num_missions']:,}\n"
            f"- Total number of weapons delivered: {summary['total_weapons']:,}\n"
        )
        # Small sample of rows
        sample_df = (                               # Small sample of rows (will go in system prompt)
            df.select(
                "MSNDATE",
                "VALID_AIRCRAFT_ROOT",
                "MFUNC_DESC",
                "NUMWEAPONSDELIVERED",
                "TGTLATDD_DDD_WGS84",
                "TGTLONDDD_DDD_WGS84",
                "MILSERVICE"
            ).head(10).to_pandas()
        )
        # Format sample as markdown table (will go in system prompt)
        sample_text = sample_df.to_markdown(index=False)

    # --------------------------------------
    # LLM SYSTEM PROMPT
    # --------------------------------------
    system_prompt = f"""
    You are an assistant helping a user explore a dataset of Vietnam War bombing missions. 
    The user is interacting with a Streamlit dashboard thta filters the data by year,
    aircraft type, and mission type.

    Here is a brief summary of the CURRENT filtered dataset:
    {summary_text}

    Here is a small sample of rows from the CURRENT filtered data:
    {sample_text}

    When you answer:
    - Speak **directly to the user** in a natural, conversational tone.
    - Use the second persion ("you") and past tense where appropriate.
    - Do **not** describe what you are about to do.
    - Do **not** say things like "we need to respond to the user" or "the assistant should".
    - Just give the final answer, as if you were chatting with the user.

    Use this information plus general knowledge of the Vietnam War to answer the user's 
    questions about trends, comparisons, and ideas for further exploration.
    If you don't have enough information to be precise, say so explicitly.
    Do NOT make up exact numeric values that aren't implied by the summary above.
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
# TITLE & YEAR FILTER
# --------------------------------------

st.title("Vietnam War Aerial Bombing")  # Dashboard title

# Filter sidebar settings
st.sidebar.header("Filters")            # Sidebar header for filters
yr_start = st.sidebar.number_input(     # Start year filter
    label="Start Year",                 # Input label
    min_value=1964,                     # Minimum year
    max_value=1973,                     # Maximum year
    value=1966)                         # Default value
yr_end   = st.sidebar.number_input(     # End year filter
    label="End Year",                   # Input label
    min_value=1964,                     # Minimum year
    max_value=1973,                     # Maximum year
    value=1968)                         # Default value


# --------------------------------------
# SQL QUERY
# --------------------------------------
#query = f"""
#SELECT MSNDATE, TGTLATDD_DDD_WGS84, TGTLONDDD_DDD_WGS84, NUMWEAPONSDELIVERED, VALID_AIRCRAFT_ROOT, MFUNC_DESC, MILSERVICE
#FROM missions
#"""

# Run the sql query - this'll create the DB if it's the first run
#df = run_query(sql=query)


# The DB read-in of our query produces strings, so we'll need to 
# re-convert MSNDATE to a Datetime object
#df = df.with_columns(
#    pl.col("MSNDATE").str.strptime(pl.Datetime, strict=False)
#)

# Filter by year range from sidebar
#df = df.filter(
#    pl.col("MSNDATE").dt.year().is_between(yr_start, yr_end)
#)

# All the above query and filtering is now handled in load_data() b/c we switched to Parquet
# instead of SQLite for simplicity.
df = load_data(yr_start, yr_end)  # Load & filter data from Parquet file

# --------------------------------------
# AIRCRAFT MULTISELECT
# --------------------------------------

# Multiselect presents all unique aircraft types from the filtered df
# Ea. aircraft type selected further filters the df to only those types
aircraft_options = (
    df.select(pl.col("VALID_AIRCRAFT_ROOT"))    # Select the aircraft column
      .drop_nulls()                             # Drop nulls
      .unique()                                 # Get unique values
      .sort("VALID_AIRCRAFT_ROOT")              # Sort the values alphabetically
      .to_series()                              # Convert to a Series
      .to_list()                                # Convert to a list (what multiselect needs)
)

selected_aircraft = st.sidebar.multiselect(     # Multiselect widget in sidebar
    "Aircraft (optional)",                      # Input label
    options=aircraft_options,                   # Options from the unique aircraft types
    default=[]                                  # Default = none selected
)

if selected_aircraft:                           # If the user selected any aircraft types
    df = df.filter(
        pl.col("VALID_AIRCRAFT_ROOT").is_in(selected_aircraft)
    )                                           # Filter the df to only those types


# --------------------------------------
# MISSION TYPE MULTISELECT - I've commented this out for now to reduce the number of filters
# --------------------------------------
#mission_type_options = (                        # Get unique mission types from the filtered df
#    df.select(pl.col("MFUNC_DESC"))             # Select the mission type column
#      .drop_nulls()                             # Drop nulls
#      .unique()                                 # Get unique values
#      .sort("MFUNC_DESC")                       # Sort the values alphabetically
#      .to_series()                              # Convert to a Series
#      .to_list()                                # Convert to a list (what multiselect needs)
#)
#selected_mission_type = st.sidebar.multiselect(     # Multiselect widget in sidebar
#    "Mission Type (optional)",                      # Input label
#    options=mission_type_options,                   # Options from the unique mission types
#    default=[]                                      # Default = none selected
#)
#if selected_mission_type:                       # If the user selected any mission types
#    df = df.filter(
#        pl.col("MFUNC_DESC").is_in(selected_mission_type)
#    )                                           # Filter the df to only those mission types



# --------------------------------------
# TIME-SERIES BARCHARTS SIDE-BY-SIDE
# --------------------------------------
col1, col2 = st.columns(2)            # Create 2 columns for side-by-side charts
with col1:                           # First column
    
    # --------------------------------------
    # NUMBER OF MISSIONS PER DAY
    # --------------------------------------

    st.markdown(                      # Show number of missions after filtering
        f"<center><b>Number of missions per day (total: {len(df):,})</b></center>",
        unsafe_allow_html=True        # Render HTML safely - not plain text
    )

    df_missions = (
        df.group_by("MSNDATE")
            .agg(pl.len().alias("num_missions"))
            .sort("MSNDATE")
    )

    missions_pd = df_missions.to_pandas().rename(
        columns={"MSNDATE": "date", "num_missions": "missions"}
    )

    missions_chart = (
        alt.Chart(missions_pd)                          # Create Altair chart from pandas df
        .mark_bar()                                     # Bar chart
        .encode(                                        # Encoding for the chart, includes...
            x=alt.X("date:T", title="Day"),             # x-axis as date (from MSNDATE datetime column)
            y=alt.Y(                                    
                "missions:Q",                           # y-axis uses the missions column as quantitative data
                title="Number of Missions",             # y-axis title
                axis=alt.Axis(format=",.0f")            # y-axis with thousands separator
            ),
            tooltip=[                                   # Tooltip on hover includes...
                alt.Tooltip("date:T", title="Date"),    # Date (temporal data type) with title
                alt.Tooltip(                            # Tooltip with thousands separator
                    "missions:Q",                       # Quantitative missions data with title
                    title="Missions",
                    format=","                          # Thousands separator
                ),
            ],
        )
    )

    st.altair_chart(missions_chart, use_container_width=True)  # Show the chart in the first column

with col2:                           # Second column
    
    # --------------------------------------
    # NUMBER OF ORDNANCE DROPPED PER DAY
    # --------------------------------------

    # Barchart title
    st.markdown(
        "<center><b>Number of ordinance dropped per day</b></center>",
        unsafe_allow_html=True          # Render HTML safely - not plain text
    )

    # Group-by to show the total num. of weapons delivered, grouped by date
    df_ts = (
        df.group_by("MSNDATE").agg(
            pl.col("NUMWEAPONSDELIVERED").cast(pl.Int64).sum()
        ).sort("MSNDATE")
    )   

    # Send the polars df to a pandas df - Streamlit plots prefer pandas
    ts_pd = df_ts.to_pandas().rename(
        columns={"MSNDATE": "date", "NUMWEAPONSDELIVERED": "weapons"}
    )

    # Altair chart with formatted tooltip - the default st barchart doesn't have nicely
    # formatted popups when you hover over the bars, so this replacement with Altair 
    # charts fixes that - even though the default barchart is an Altair chart 
    chart = (
        alt.Chart(ts_pd)                                # Create Altair chart from pandas df
        .mark_bar()                                     # Bar chart
        .encode(                                        # Encoding for the chart, includes...
            x=alt.X("date:T", title="Day"),             # x-axis as date (from MSNDATE datetime column)
            y=alt.Y(                                    
                "weapons:Q",                            # y-axis uses the weapons column as quantitative data
                title="Number of Ordnance Delivered",   # y-axis title
                axis=alt.Axis(format=",.0f")            # y-axis with thousands separator
            ),
            tooltip=[                                   # Tooltip on hover includes...
                alt.Tooltip("date:T", title="Date"),    # Date (temporal data type) with title
                alt.Tooltip(                            # Tooltip with thousands separator
                    "weapons:Q",                        # Quantitative weapons data with title
                    title="Weapons Delivered",
                    format=","                          # Thousands separator
                ),
            ],
        )
    )

    # Plot the visualization in the dashboard
    # Show the Altair chart, use full container width
    st.altair_chart(                
        chart, 
        use_container_width=True
    )    


# --------------------------------------
# MILSERVICE DOUGHNUT & MISSION TYPES SIDE-BY-SIDE
# --------------------------------------
col1, col2 = st.columns(2)            # Create 2 columns for side-by-side charts
with col1:                           # First column

    # --------------------------------------
    # MILSERVICE DOUGHNUT CHART
    # --------------------------------------
    st.markdown(                      # Show milservice doughnut chart title
        "<center><b>Missions by military service</b></center>",
        unsafe_allow_html=True        # Render HTML safely - not plain text
    )
    df_mislservice = (          # Group by milservice & count
        df.group_by("MILSERVICE")).agg(
            pl.len().alias("num_missions")
        ).sort("num_missions", descending=True)
    df_mislservice = df_mislservice.with_columns(   # Calculate percent of total missions
        (pl.col("num_missions") / pl.col("num_missions").sum()).alias("percent")
    )
    milservice_pd = df_mislservice.to_pandas().rename(
        columns={"MILSERVICE": "milservice", "num_missions": "missions", "percent": "percent(missions)"}
    )

    service_colors = {
    "KAF": [255, 255, 255, 160],
    "OTHER": [200, 200, 200, 160],
    "RAAF": [80, 200, 120, 160],
    "RLAF": [234, 155, 0, 160],
    "USAF": [0, 114, 178, 160],
    "USN":  [0, 0, 255, 160],
    "USMC": [201, 41, 35, 160],
    "USA": [110, 161, 113, 160],
    "VNAF": [253, 205, 17, 160],
    }

    # Build domain + range from the dict
    # (limit to services that actually appear in the data, just in case)
    services_in_data = milservice_pd['milservice'].unique().tolist()  # Get list of services in the data
    domain = services_in_data
    range_colors = []
    for svc in services_in_data:
        r, g, b, a = service_colors.get(svc, [200, 200, 200, 160])  # Default gray if not found
        range_colors.append(f"rgba({r},{g},{b},{a/255})")
    
    milservice_chart = (         # Altair doughnut chart
        alt.Chart(milservice_pd)  # Create Altair chart from pandas df
        .mark_arc(innerRadius=50)    # Doughnut chart with inner radius
        .encode(                   # Encoding for the chart, includes...
            theta=alt.Theta(       # Angle uses number of missions
                "missions:Q",      # Quantitative data type
                title="Number of Missions"
            ),
            color=alt.Color(       # Color uses milservice
                "milservice:N",    # Nominal data type
                title="Military Service",
                scale=alt.Scale(
                    domain=domain,
                    range=range_colors
                )
            ),
            tooltip=[              # Tooltip on hover includes...
                alt.Tooltip(       # Tooltip with thousands separator
                    "missions:Q",  # Quantitative missions data with title
                    title="Missions",
                    format=","     # Thousands separator
                ),
                alt.Tooltip(       # Tooltip for milservice
                    "milservice:N",
                    title="Military Service"
                ),
                alt.Tooltip(       # Tooltip for percentage of total missions
                    "percent(missions):Q",  # Percentage of missions
                    title="Percentage of Total Missions",
                    format=".2%"        # Percentage format with 2 decimal places
                )
            ]
        )
    )
    st.altair_chart(milservice_chart, use_container_width=True)  # Display the chart in Streamlit
    
with col2:                           # Second column
    
    # --------------------------------------
    # MISSION TYPES HORIZONTAL BAR CHART
    # --------------------------------------
    st.markdown(                      # Show mission types bar chart title
        "<center><b>Top 10 mission types</b></center>",
        unsafe_allow_html=True        # Render HTML safely - not plain text
    )
    df_mission_types = (              # Group by mission type & count
        df.group_by("MFUNC_DESC")
          .agg(pl.len().alias("num_missions"))
          .sort("num_missions", descending=True)
    ).head(10)                     # Take top 10 mission types
    mission_types_pd = df_mission_types.to_pandas().rename(
        columns={"MFUNC_DESC": "mission_type", "num_missions": "missions"}
    )
    mission_types_chart = (           # Altair horizontal bar chart
        alt.Chart(mission_types_pd)    # Create Altair chart from pandas df
        .mark_bar()                    # Bar chart
        .encode(                       # Encoding for the chart, includes...
            y=alt.Y(                   # y-axis uses mission type
                "mission_type:N",      # Nominal data type
                title="Mission Type",  # y-axis title
                sort='-x'              # Sort by x-axis values descending
            ),
            x=alt.X(                   # x-axis uses number of missions
                "missions:Q",          # Quantitative data type
                title="Number of Missions",  # x-axis title
                axis=alt.Axis(format=",.0f") # x-axis with thousands separator
            ),
            tooltip=[                  # Tooltip on hover includes...
                alt.Tooltip(           # Tooltip with thousands separator
                    "missions:Q",      # Quantitative missions data with title
                    title="Missions",
                    format=","         # Thousands separator
                )
            ]
        )
    )
    st.altair_chart(mission_types_chart, use_container_width=True)  # Display the chart in Streamlit



# --------------------------------------
# MAP OF FILTERED DATA
# --------------------------------------

# To add toggleable layers to our map, we'll use pydeck which is what Streamlit's st.map()
# is built on. 
# Map title
st.markdown(
    "<center><b>Target locations</b></center>",
    unsafe_allow_html=True
)



# Streamlit maps require lat/lon columns named "lat" & "lon" - select & rename those cols
df_map = df.select(
    pl.col("TGTLATDD_DDD_WGS84").alias("lat"),
    pl.col("TGTLONDDD_DDD_WGS84").alias("lon"),
    pl.col("MILSERVICE"),               # Include milservice for potential future use
    pl.col("VALID_AIRCRAFT_ROOT"),  # Include aircraft for potential future use
    pl.col("MFUNC_DESC")          # Include mission type for potential future use
).drop_nulls()                                  # Drop nulls to avoid errors in the map

map_pd = df_map.to_pandas()          # Convert to pandas for Streamlit map

services = sorted(map_pd["MILSERVICE"].dropna().unique())  # Get unique milservices in the filtered data

selected_services = st.multiselect(          # Multiselect for milservices to show on the map
    "Show services on map (leave empty to show all):",
    options=services,
    default=services
)

#legend_services = selected_services if selected_services else services  # Services to show in legend

#st.markdown("**Legend:**")  # Legend title
#legend_html = ""            # Initialize legend HTML string
#for svc in legend_services: # For each milservice to show in legend
#    color = service_colors.get(svc, [200, 200, 200, 160])  # Get color or default gray
#    r, g, b, _ = color
#    legend_html += f"""
#    <div style="display: flex; align-items: center; margin-bottom: 4px;">
#        <span style="
#            display: inline-block;
#            width: 14px;
#            height: 14px;
#            margin-right: 6px;
#            border-radius: 2px;
#            background-color: rgba({r}, {g}, {b}, 0.6);
#        "></span>
#        <span>{svc}</span>
#    </div>
#    """
#st.markdown(legend_html, unsafe_allow_html=True)  # Display the legend in Streamlit

layers = []                                        # List to hold map layers
for svc in selected_services:                       # For each milservice
    svc_data = map_pd[map_pd["MILSERVICE"] == svc]  # Filter data for this milservice
    color = service_colors.get(svc, [100, 100, 100, 160])  # Get color or default gray

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",                    # Layer type
            data=svc_data,                         # Data for this layer
            get_position=["lon", "lat"],          # Position from lon/lat columns
            get_fill_color=color,                  # Color for this milservice
            get_radius=500,                        # Radius of points in meters
            pickable=True,                        # Enable tooltips on hover
            auto_highlight=True,                  # Highlight on hover
            name=svc                              # Layer name for legend
        )
    )

view_state = pdk.ViewState(                    # Initial view settings
    latitude=map_pd["lat"].mean(),              # Center latitude
    longitude=map_pd["lon"].mean(),             # Center longitude
    zoom=4,                                     # Zoom level
    pitch=0                                     # No tilt
)

r = pdk.Deck(                                 # Create the pydeck Deck object
    map_style="dark",                           # Dark map style
    layers=layers,                            # Add our layers
    initial_view_state=view_state,            # Set the initial view
    tooltip={                                 # Tooltip settings
        "text": "Service: {MILSERVICE}\n"
                "Aircraft: {VALID_AIRCRAFT_ROOT}\n"
                "Mission Type: {MFUNC_DESC}"
    }
)

st.pydeck_chart(r)                           # Show the pydeck map in Streamlit


# Default map view settings - I've commented out for the more detailed pydeck map above
#st.map(
#    df_map.to_pandas(),                         # Convert to pandas for Streamlit map
#    use_container_width=True                    # Use full container width
#)






# --------------------------------------
# CHAT ASSISTANT UI
# --------------------------------------
st.markdown("---")                          # Horizontal grey line to separate sections
st.subheader("Ask the data assistant")      # Chat assistant subheader

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
        "Ask a question about the data, filters, aircraft, etc."
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
                    assistant_reply = run_llm(                  # Call our LLM helper function
                        user_msg=user_msg,                      # User's message
                        df=df,                                  # Current dataframe with user filters
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