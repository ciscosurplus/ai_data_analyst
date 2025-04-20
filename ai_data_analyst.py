import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
import re
import os
from pathlib import Path

# Function to save API key to file
def save_api_key(api_key):
    # Create directory if it doesn't exist
    config_dir = Path.home() / ".data_analyst_agent"
    config_dir.mkdir(exist_ok=True)
    
    # Save the API key to a file
    config_file = config_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump({"openai_key": api_key}, f)

# Function to load API key from file
def load_api_key():
    config_file = Path.home() / ".data_analyst_agent" / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            try:
                config = json.load(f)
                return config.get("openai_key")
            except json.JSONDecodeError:
                return None
    return None

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app
st.title("📊 Data Analyst Agent")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Try to load saved API key
if "openai_key" not in st.session_state:
    saved_key = load_api_key()
    if saved_key:
        st.session_state.openai_key = saved_key

# Sidebar for API keys and clear button
with st.sidebar:
    st.header("API Keys")
    
    # Show current API key status
    if "openai_key" in st.session_state and st.session_state.openai_key:
        st.success("API key is saved!")
        if st.button("Change API Key"):
            st.session_state.change_key = True
    else:
        st.session_state.change_key = True
    
    # Display input field if key needs to be entered or changed
    if st.session_state.get("change_key", True):
        openai_key = st.text_input("Enter your OpenAI API key:", type="password")
        if openai_key:
            st.session_state.openai_key = openai_key
            save_api_key(openai_key)
            st.success("API key saved!")
            st.session_state.change_key = False
            st.rerun()
        else:
            st.warning("Please enter your OpenAI API key to proceed.")
    
    # Add a clear button to refresh the chat
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Configure the semantic model with the temporary file path
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }
        
        # Initialize the DuckDbAgent for SQL query generation
        duckdb_agent = DuckDbAgent(
            model=OpenAIChat(api_key=st.session_state.openai_key, model_name="gpt-4.1-mini"),
            semantic_model=json.dumps(semantic_model),
            markdown=True,
            add_history_to_messages=False,  # Disable chat history
            followups=False,  # Disable follow-up queries
            read_tool_call_history=False,  # Disable reading tool call history
            system_prompt="You are an expert data analyst. Generate SQL queries to solve the user's query. Return only the SQL query, enclosed in ```sql ``` and give the final answer.",
        )
        
        # Initialize code storage in session state
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None
        
        # Display chat history
        st.subheader("Chat History")
        for i, (query, response) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {query}")
            st.markdown(f"**AI:** {response}")
            st.markdown("---")
        
        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")
        
        # Add info message about terminal output
        st.info("💡 Check your terminal for a clearer output of the agent's response")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get the response from DuckDbAgent
                        response1 = duckdb_agent.run(user_query)

                        # Extract the content from the RunResponse object
                        if hasattr(response1, 'content'):
                            response_content = response1.content
                        else:
                            response_content = str(response1)
                        
                        # Call print_response for terminal output
                        duckdb_agent.print_response(
                            user_query,
                            stream=True,
                        )

                    # Add the query and response to chat history
                    st.session_state.chat_history.append((user_query, response_content))

                    # Display the response in Streamlit
                    st.markdown("**You:** " + user_query)
                    st.markdown("**AI:** " + response_content)
                    
                    # Clear the text area after submission
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")
