import os
import openai
from googleapiclient.discovery import build
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Set API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
search_engine_id = os.getenv('SEARCH_ENGINE_ID')  # Ensure this is set in your .env file

# Initialize LangChain with OpenAI
llm = ChatOpenAI(model_name="gpt-4",api_key=openai_api_key)
prompt_template = PromptTemplate(
    input_variables=["topic", "context"],
    template="You are an expert programmer. Generate a detailed script and relevant code on {topic} using the following information for context and guidance:\n{context}"
)

chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_script(topic, context):
    result = chain.run(topic=topic, context=context, max_tokens=8192)
    return result.strip()

def google_search(query):
    service = build("customsearch", "v1", developerKey=google_api_key)
    res = service.cse().list(q=query, cx=search_engine_id).execute()
    return res['items']

def fetch_search_data(topic):
    results = google_search(topic)
    context = ""
    for item in results:
        context += f"{item['title']}\n{item['snippet']}\n{item['link']}\n\n"
    return context

def generate_detailed_script(topic):
    context = fetch_search_data(topic)
    script = generate_script(topic, context)
    return script

# Streamlit interface
st.title("Script Generator")

topic = st.text_input("Enter the topic for the script:")

if st.button("Generate Script"):
    if topic:
        with st.spinner("Generating script..."):
            script = generate_detailed_script(topic)
            st.success("Script generated successfully!")
            st.text_area("Generated Script", value=script, height=400)
    else:
        st.error("Please enter a topic to generate the script.")