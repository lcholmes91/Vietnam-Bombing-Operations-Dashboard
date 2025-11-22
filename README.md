# Vietnam-Bombing-Operations-Dashboard

![Dashboard GIF](/Figures/dashboard.gif)

This repository produces an [interactive dashboard](https://vietnam-bombing-operations-dashboard-djaxjt64vgos7rfgxtnhm4.streamlit.app/) that visualizes bombing operations conducted by the United States & its allies during the Vietnam War. The dashboard provides insights into the scale, frequency, and geographical distribution of bombing missions over time. It also allows users to engage an LLM chatbot interface to ask questions about the dataset.

## Directory Structure
```
. 
├── .streamlit/         # config files for Streamlit dashboard
├── Data/               # Dataset & reference docs
├── Figures/            # Images and visualizations
├── Intro.py            # Page 1 (aka Entrypoint) of the Streamlit dashboard
├── pages/              # Page 2 of the Streamlit dashboard
├── README.md           # This file
├── requirements.txt    # Required dependencies for the Streamlit dashboard
└── runtime.txt         # Streamlit runtime environment
```

## Description of the Dataset
The [THOR Vietnam Bombing Operations](https://www.kaggle.com/datasets/usaf/vietnam-war-bombing-operations?select=THOR_Vietnam_Bombing_Operations.csv) dataset is a database of aerial bombings from the Vietnam War from 1964 - 1972. All told, the dataset totals 1.63GB, and it consists of four items: 
1. An aircraft glossary explaining the models of U.S. aircraft used during Vietnam War
2. A **bombing operation** .csv file which contains each mission’s 5Ws
3. A glossary of munitions used by the aircraft
4. An explanatory dictionary for THOR data

## How to Acquire the Dataset
Here are two ways to acquire the dataset:
1. You can pull the dataset directly into a DataFrame by reading in the *THOR_Vietnam_Bombing_Operations.parquet* file using the code provided below. This *.parquet* file is in the Data folder in this repo. 
```python
df_parquet = pl.read_parquet("Data/THOR_Vietnam_Bombing_Operations.parquet")
```
2. A comma-separted values file of the dataset, *THOR_Vietnam_Bombing_Operations.csv*, is hosted on Kaggle. To download the dataset from Kaggle, you will need to create a Kaggle account if you do not already have one. After creating an account, follow these steps:
   - Go to the [Kaggle Vietnam War Bombing Operations Dataset page](https://www.kaggle.com/datasets/usaf/vietnam-war-bombing-operations?select=THOR_Vietnam_Bombing_Operations.csv).
   - Click on the "Download" button to download the dataset as a ZIP file.
   - Extract the contents of the ZIP file to access the *THOR_Vietnam_Bombing_Operations.csv* file.

## How to Use the Dashboard
The dashboard consists of two main pages:

1. **Introduction Page**: This page provides an overview of the Vietnam War bombing operations dataset.

2. **Descriptive Statistics Page**: This page offers various visualizations and statistics about the bombing operations, including trends over time, geographical distribution, and types of munitions used.

Users can interact with the visualizations by selecting different filters and options available in the sidebar. Additionally, LLM chatbot interface allows users to ask questions about the currently filtered dataset or the conflict in general.

## How to Run the Streamlit Dashboard Locally
To run the Streamlit dashboard locally, follow these steps:
1. Clone this repository to your local machine using the command:
   ```bash
   git clone <repository_url>
   ```
2. For a stable experience, activate a virtual environment with the same dependencies in the requirements.txt file and the same Python version specified in the runtime.txt file.
    
3. Navigate to the root directory of the cloned repository:
```bash
cd <path-to-cloned-repo>
```
    
4. Run the following command to start the Streamlit dashboard:
```bash
streamlit run Intro.py
```
Under the hood, Streamlit will launch a local web client to host the dashboard, with Intro.py as the entry point server. Streamlit will automatically detect and include the other pages in the *pages* directory, in the order of their filenames.

The dashboard should open in your default web browser. If it does not, you can access it by navigating to http://localhost:8501 in your web browser.

To set up the LLM chatbot interface for your local dashboard, you will need to register for an LLM API key. I used NVIDIA's [Qwen3-Next-80B-A3B-Instruct](https://build.nvidia.com/qwen/qwen3-next-80b-a3b-instruct/modelcard) model, which is what you'll interact with if you open the [cloud-hosted dashboard](https://vietnam-bombing-operations-dashboard-djaxjt64vgos7rfgxtnhm4.streamlit.app/) I've set up. The [NVIDIA build](https://build.nvidia.com/) platform hosts various other LLMs to choose from. Once you've registered for your LLM's API key, fill in the API and model information into the provided .streamlit/secrets.toml.example file, and rename it to .streamlit/secrets.toml. The chatbot interface will then be functional within the dashboard. You can alter the system and user prompts in the Intro.py and 1_Descriptive_Statistics.py files to customize the chatbot's behavior. 

## Acknowledgements
This project was completed as part of coursework for the Naval Postgraduate School's Operations Research curriculum. Various online resources were referenced for this project, including the [THOR dataset](https://www.kaggle.com/datasets/usaf/vietnam-war-bombing-operations?select=THOR_Vietnam_Bombing_Operations.csv) on Vietnam War bombing operations, the [Streamlit documentation](https://docs.streamlit.io/) for dashboard visualization techniques, and [NVIDIA's build platform](https://build.nvidia.com/) for LLM model descriptions and recommended usage. I would like to acknowledge the significant contributions of Sean Pinzini and Kyle Ward for their collaborative assistance in data acquisition and analysis on this project. ChatGPT-5.1 and Microsoft Copilot were used to assist with code generation and debugging; any errors are my own.