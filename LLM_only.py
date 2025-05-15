import streamlit as st
import requests
from Bio import Entrez, Medline
import csv
import json
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import pandas as pd
from collections import Counter
import difflib

# Page configuration
st.set_page_config(
    page_title="Drug Repurposing Analysis Tool",
    page_icon="💊",
    layout="wide",
)

# Load environment variables
load_dotenv()

@st.cache_resource
def configure_entrez(email):
    Entrez.email = email

@st.cache_resource
def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.session_state.get('api_key', '')
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    return None

# Build PubMed query

def build_pubmed_query(primary_disease, comorbidity):
    primary_query = f'"{primary_disease}"[MeSH Terms] OR "{primary_disease}"[Title/Abstract]'
    comorbidity_query = f'"{comorbidity}"[MeSH Terms] OR "{comorbidity}"[Title/Abstract]'
    return f"(({primary_query}) AND ({comorbidity_query})) AND (\"drug therapy\"[Subheading] OR \"pharmacology\"[Subheading] OR \"therapeutic use\"[Subheading] OR \"treatment\"[Title/Abstract]) AND humans[Filter]"

# Fetch PubMed abstracts

def fetch_pubmed_abstracts(query, max_results=25):
    with st.spinner('Fetching articles from PubMed...'):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            pmids = record["IdList"]
            if not pmids:
                return []
            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
            return list(Medline.parse(handle))
        except Exception as e:
            st.error(f"Error fetching PubMed data: {str(e)}")
            return []

# Use Gemini to extract relationships

def extract_relationships(abstracts, primary_disease, comorbidity):
    model = configure_gemini()
    if not model:
        st.error("Gemini API key not configured properly")
        return None
    full_text = "\n".join([
        f"Title: {record.get('TI', '')}\nAbstract: {record.get('AB', '')}\n"
        for record in abstracts if record.get('AB')
    ])
    prompt = f"""Analyze the following collection of medical abstracts and extract relationships between diseases, 
    comorbidities, and drugs. Only use the abstracts provided.

    **Your Task**:
    - Identify the **primary disease**: {primary_disease}.
    - Identify the **comorbidity** being analyzed: {comorbidity}.
    - List **drugs used to treat the primary disease**.
    - List **drugs used to treat the comorbidity**.
    - Identify **drugs that are effective for both conditions**.
    - For each drug that treats both conditions, provide:
      1. A score from 1-10 on potential for repurposing (10 being highest)
      2. Evidence from literature for its dual efficacy
      3. Mechanism of action explaining why it might work for both conditions
      4. Potential molecular targets relevant to both conditions (if mentioned)
    
    **Output Format (JSON)**:
    {{
        "primary_disease": "{primary_disease}",
        "comorbidity": "{comorbidity}",
        "drugs_primary_disease": ["Drug1", "Drug2"],
        "drugs_comorbidity": ["DrugA", "DrugB"],
        "shared_treatments": [
            {{
                "drug": "Drug Name",
                "repurposing_score": 8, 
                "primary_disease_treatment": true,
                "comorbidity_treatment": true,
                "evidence": "Brief evidence from literature",
                "mechanism_of_action": "Inhibits X receptor, reducing Y",
                "molecular_targets": ["target1", "target2"]
            }}
        ],
        "explanation": "Overall explanation for drugs that are effective for both conditions."
    }}

    **Medical Abstracts**: 
    {full_text}
    """
    try:
        with st.spinner('Analyzing abstracts with AI...'):
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.3))
            text_response = re.sub(r'```json\s*', '', response.text)
            text_response = re.sub(r'```\s*', '', text_response).strip()
            return json.loads(text_response)
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return None

# Function to save data to CSV
def save_to_csv(data, primary_disease, comorbidity):
    if not data or 'shared_treatments' not in data:
        return None
    
    filename = f"{primary_disease}_{comorbidity}_repurposing_results.csv"
    
    # Create DataFrame for shared treatments
    if data['shared_treatments']:
        df = pd.DataFrame(data['shared_treatments'])
        df.to_csv(filename, index=False)
        return filename
    return None

# Function to visualize drug rankings (without bar chart)
def visualize_drug_rankings(data):
    if not data or 'shared_treatments' not in data or not data['shared_treatments']:
        st.info("No shared treatments found to visualize.")
        return
    
    shared_treatments = data['shared_treatments']
    
    # Sort by repurposing score
    sorted_drugs = sorted(shared_treatments, key=lambda x: x.get('repurposing_score', 0), reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame(sorted_drugs)
    
    # Show detailed table
    st.subheader("Detailed Drug Information")
    
    # Reformat the DataFrame for display
    display_df = df[['drug', 'repurposing_score', 'mechanism_of_action', 'evidence', 'molecular_targets']].copy()
    display_df.columns = ['Drug', 'Repurposing Score', 'Mechanism of Action', 'Evidence', 'Potential Molecular Targets']
    
    st.dataframe(display_df)

def main():
    st.title("💊 LLM-Based Drug Repurposing Analysis Tool")
    st.write("Discover potential drug repurposing opportunities by analyzing comorbidities using literature.")

    with st.sidebar:
        st.header("Settings")
        user_email = st.text_input("Email for PubMed API", value="your_email@example.com")
        configure_entrez(user_email)
        api_key = st.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        if api_key:
            st.session_state['api_key'] = api_key
            genai.configure(api_key=api_key)

    col1, col2 = st.columns(2)
    with col1:
        primary_disease = st.text_input("Primary Disease", value="Cirrhosis of liver")
    with col2:
        comorbidity = st.text_input("Comorbidity", value="Chronic Kidney Disease")

    max_results = st.slider("Maximum number of articles to analyze", 5, 100, 25)

    if st.button("Search & Analyze", type="primary"):
        if not primary_disease or not comorbidity:
            st.warning("Please enter both primary disease and comorbidity.")
            return

        query = build_pubmed_query(primary_disease, comorbidity)
        abstracts = fetch_pubmed_abstracts(query, max_results)

        if not abstracts:
            st.warning(f"No abstracts found for {primary_disease} and {comorbidity}.")
            return

        st.success(f"Found {len(abstracts)} relevant articles.")

        results = extract_relationships(abstracts, primary_disease, comorbidity)

        if results:
            st.header(f"Analysis Results: {primary_disease} + {comorbidity}")
            st.subheader("Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Primary Disease Drugs", len(results.get('drugs_primary_disease', [])))
            with col2:
                st.metric("Comorbidity Drugs", len(results.get('drugs_comorbidity', [])))

            st.subheader("Shared Treatments")
            shared_count = len(results.get('shared_treatments', []))
            if shared_count > 0:
                st.success(f"Found {shared_count} potential drugs for repurposing!")
                visualize_drug_rankings(results)
                csv_file = save_to_csv(results, primary_disease, comorbidity)
                if csv_file:
                    with open(csv_file, 'rb') as f:
                        st.download_button(
                            label="Download Results as CSV",
                            data=f,
                            file_name=csv_file,
                            mime="text/csv"
                        )
            else:
                st.info("No shared treatments found between the diseases based on the analyzed abstracts.")

            if 'explanation' in results:
                st.subheader("General Analysis")
                st.write(results['explanation'])

            st.subheader("Raw JSON Output")
            st.json(results)

if __name__ == "__main__":
    main()