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
import difflib
from collections import defaultdict

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

@st.cache_data
def load_kg():
    kg_df = pd.read_csv("kg.csv", low_memory=False)
    kg_df['x_type'] = kg_df['x_type'].str.lower()
    kg_df['y_type'] = kg_df['y_type'].str.lower()
    kg_df['relation'] = kg_df['relation'].str.lower()
    kg_df['display_relation'] = kg_df['display_relation'].str.lower()
    return kg_df

# Fuzzy match helper for KG
def get_closest_disease_name_kg(disease_name, kg_df, min_similarity=0.8):
    disease_name_lower = disease_name.lower()
    disease_names = kg_df[kg_df['x_type'] == 'disease']['x_name'].dropna().str.lower().unique()
    
    # Manual best match using similarity ratio
    best_match = None
    best_score = 0

    for candidate in disease_names:
        score = difflib.SequenceMatcher(None, disease_name_lower, candidate).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= min_similarity:
        return best_match
    else:
        return None


# Gene retrieval for KG
def get_genes_for_disease_kg(disease_name, kg_df):
    matched_name = get_closest_disease_name_kg(disease_name, kg_df)
    if not matched_name:
        return []
    return kg_df[
        (kg_df['x_name'].str.lower() == matched_name) &
        (kg_df['x_type'] == 'disease') &
        (kg_df['y_type'] == 'gene/protein') &
        (kg_df['display_relation'].isin({'associated with', 'expression present', 'expression absent'}))
    ]['y_name'].dropna().unique().tolist()

# Drug retrieval targeting genes for KG
def get_drugs_targeting_genes_kg(genes, kg_df):
    return kg_df[
        (kg_df['y_name'].isin(genes)) &
        (kg_df['x_type'] == 'drug') &
        (kg_df['y_type'] == 'gene/protein') &
        (kg_df['display_relation'].isin({'target', 'enzyme', 'carrier', 'transporter', 'ppi'}))
    ][['x_name', 'y_name']].drop_duplicates()

# Count genes targeted by each drug for KG
def count_genes_targeted_kg(drug_targets):
    drug_gene_counts = defaultdict(set)
    for _, row in drug_targets.iterrows():
        drug = row['x_name']
        gene = row['y_name']
        drug_gene_counts[drug].add(gene)
    return {drug: len(genes) for drug, genes in drug_gene_counts.items()}

# Rank drugs by the number of genes targeted for KG
def rank_drugs_by_gene_count_kg(drug_gene_counts):
    return sorted(drug_gene_counts.items(), key=lambda item: item[1], reverse=True)

# Function to visualize drug targets
def visualize_drug_targets(ranked_drugs, drug_targets, disease_name):
    if not ranked_drugs:
        st.info(f"No drugs found targeting genes associated with {disease_name}.")
        return

    st.subheader(f"Top Drugs Targeting Genes Associated with {disease_name} (KG Analysis)")
    for drug, count in ranked_drugs[:10]:
        targeted_genes = drug_targets[drug_targets['x_name'] == drug]['y_name'].unique().tolist()
        st.markdown(f"**{drug}**: Targets **{count}** genes - {', '.join(targeted_genes)}")

# Build PubMed query
def build_pubmed_query(primary_disease, comorbidity):
    primary_query = f'"{primary_disease}"[MeSH Terms] OR "{primary_disease}"[Title/Abstract]'
    comorbidity_query = f'"{comorbidity}"[MeSH Terms] OR "{comorbidity}"[Title/Abstract]'
    return f"(({primary_query}) AND ({comorbidity_query})) AND (\"drug therapy\"[Subheading] OR \"pharmacology\"[Subheading] OR \"therapeutic use\"[Subheading] OR \"treatment\"[Title/Abstract]) AND humans[Filter]"

# Fetch PubMed abstracts
def fetch_pubmed_abstracts(query, max_results=25):
    try:
        with st.spinner('Searching PubMed...'):
            search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(search_handle)
            pmids = record["IdList"]

        if not pmids:
            return []

        abstracts = []
        with st.spinner('Fetching abstracts from PubMed...'):
            # Progress bar
            progress_bar = st.progress(0)
            chunk_size = max(1, len(pmids) // 10)  # divide into 10 chunks

            for i in range(0, len(pmids), chunk_size):
                batch_ids = pmids[i:i + chunk_size]
                fetch_handle = Entrez.efetch(db="pubmed", id=",".join(batch_ids), rettype="medline", retmode="text")
                batch_records = list(Medline.parse(fetch_handle))
                abstracts.extend(batch_records)
                progress_bar.progress(min(1.0, (i + chunk_size) / len(pmids)))

            progress_bar.empty()
            return abstracts
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
    - Identify **drugs that are effective for both conditions**.\n    - For each drug that treats both conditions, provide:
      1. A score from 1-10 on potential for repurposing (10 being highest)(If the drug is not explicitly mentioned in the literature, but is biologically plausible based on the knowledge graph, assign a score between 3–6 based on inferred mechanism.)
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
            response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
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
    if data['shared_treatments']:
        df = pd.DataFrame(data['shared_treatments'])
        df.to_csv(filename, index=False)
        return filename
    return None

def load_fda_approved_drugs(txt_path):
    approved_drugs = set()
    with open(txt_path, 'r', encoding='utf-8') as file:
        next(file)  # Skip header
        for line in file:
            fields = line.strip().split('~')
            if len(fields) >= 1:
                drug = fields[0].strip().upper()
                if drug:
                    approved_drugs.add(drug)
    return approved_drugs


# Function to visualize drug rankings
def visualize_drug_rankings(data, approved_drug_names=None):
    if not data or 'shared_treatments' not in data or not data['shared_treatments']:
        st.info("No shared treatments found to visualize.")
        return

    shared_treatments = data['shared_treatments']
    sorted_drugs = sorted(shared_treatments, key=lambda x: x.get('repurposing_score', 0), reverse=True)
    df = pd.DataFrame(sorted_drugs)
    display_df = df[['drug', 'repurposing_score', 'mechanism_of_action', 'evidence', 'molecular_targets']].copy()
    display_df.columns = ['Drug', 'Repurposing Score', 'Mechanism of Action', 'Evidence', 'Potential Molecular Targets']
    st.subheader("Detailed Drug Information")
    st.dataframe(display_df)

    if approved_drug_names:
        approved = []
        for drug in display_df['Drug']:
            drug_upper = str(drug).upper()
            for approved_name in approved_drug_names:
                if approved_name in drug_upper or drug_upper in approved_name:
                    approved.append(drug)
                    break

        if approved:
            st.subheader("FDA Approved Drugs:")
            st.write("Others that are not shown might be drug groups/experimental/investigational/approved in other countries")
            for drug in approved:
                st.markdown(f"- {drug}")
        else:
            st.info("No FDA approved drugs found among the results.")

def main():
    st.title("💊 Drug Repurposing Analysis Tool")
    st.write("Discover potential drug repurposing opportunities by analyzing literature and knowledge graphs.")
    FDA_TXT_PATH = "products.txt"  # replace with your file name
    approved_drug_names = load_fda_approved_drugs(FDA_TXT_PATH)

    with st.sidebar:
        st.header("Settings")
        user_email = st.text_input("Email for PubMed API", value="your_email@example.com")
        configure_entrez(user_email)
        api_key = st.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        if api_key:
            st.session_state['api_key'] = api_key
            genai.configure(api_key=api_key)

    # with st.sidebar.expander("📋 Can't find your disease?"):
    #     st.markdown("### Look here for the closest match")
    #     st.markdown(
    #         '[🔍 Click to view full `kg.csv` file](kg.csv){:target="_blank"}',
    #         unsafe_allow_html=True
    #     )

    col1, col2 = st.columns(2)
    with col1:
        primary_disease = st.text_input("Primary Disease", value="Heart Disease")
    with col2:
        comorbidity = st.text_input("Comorbidity", value="Diabetes Mellitus")

    max_results = st.slider("Maximum number of articles to analyze", 5, 1000, 50, step=25)
    analysis_mode = st.radio("Analysis Mode", ["LLM Only", "KG Only", "LLM + KG"], index=0)

    # # Load DrugBank data
    # drugbank_file = "drugbank-v2.9.tsv"  # Replace with the actual path to your drugbank.csv file
    # if os.path.exists(drugbank_file):
    #     drugbank_df = pd.read_csv(drugbank_file, sep="\t")
    # else:
    #     drugbank_df = None
    #     st.warning(f"DrugBank file not found at {drugbank_file}.  Approved drug filtering will be skipped.")

    if st.button("Search & Analyze", type="primary"):
        if not primary_disease or not comorbidity:
            st.warning("Please enter both primary disease and comorbidity.")
            return
        
        if analysis_mode == "LLM Only":
            query = build_pubmed_query(primary_disease, comorbidity)
            abstracts = fetch_pubmed_abstracts(query, max_results)

            if not abstracts:
                st.warning(f"No abstracts found for {primary_disease} and {comorbidity}.")
                return

            st.success(f"Found {len(abstracts)} relevant articles.")
            results = extract_relationships(abstracts, primary_disease, comorbidity)
            
            if results:
                st.header(f"Analysis Results (LLM Only): {primary_disease} + {comorbidity}")
                st.subheader("Shared Treatments")
                shared_count = len(results.get('shared_treatments', []))
                if shared_count > 0:
                    st.success(f"Found {shared_count} potential drugs for repurposing!")
                    visualize_drug_rankings(results, approved_drug_names) 
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

        elif analysis_mode == "KG Only":
            kg_df = load_kg()
            matched_primary_disease = get_closest_disease_name_kg(primary_disease, kg_df)
            matched_comorbidity = get_closest_disease_name_kg(comorbidity, kg_df)

            st.info(f"Matched Primary Disease: {matched_primary_disease}")
            st.info(f"Matched Comorbidity: {matched_comorbidity}")

            if matched_primary_disease and matched_comorbidity:
                genes_primary = get_genes_for_disease_kg(matched_primary_disease, kg_df)
                genes_comorbidity = get_genes_for_disease_kg(matched_comorbidity, kg_df)
                shared_genes = list(set(genes_primary) & set(genes_comorbidity))

                st.info(f"Genes for Primary Disease: {', '.join(genes_primary) if genes_primary else 'No genes found.'}")
                st.info(f"Genes for Comorbidity: {', '.join(genes_comorbidity) if genes_comorbidity else 'No genes found.'}")
                st.info(f"Shared Genes: {', '.join(shared_genes) if shared_genes else 'No shared genes found.'}")

                st.header(f"Drug Repurposing Analysis (KG Only): {primary_disease} + {comorbidity}")

                if shared_genes:
                    drug_targets_shared = get_drugs_targeting_genes_kg(shared_genes, kg_df)
                    drug_targets_shared = drug_targets_shared[~drug_targets_shared['x_name'].str.contains(r'(fibroblast|keratinocyte|neonatal|ovine|recombinant)', case=False)]
                    drug_shared_gene_counts = count_genes_targeted_kg(drug_targets_shared)
                    ranked_drugs_shared = rank_drugs_by_gene_count_kg(drug_shared_gene_counts)
                    st.subheader("Top Repurposed Drug Candidates (Ranked by Number of Shared Genes Targeted)")
                    visualize_drug_targets(ranked_drugs_shared, drug_targets_shared, "Shared Genes")

                elif genes_primary and not genes_comorbidity:
                    st.info(f"No shared genes found. Listing drugs targeting genes associated with {primary_disease}.")
                    drug_targets_primary = get_drugs_targeting_genes_kg(genes_primary, kg_df)
                    drug_targets_primary = drug_targets_primary[~drug_targets_primary['x_name'].str.contains(r'(fibroblast|keratinocyte|neonatal|ovine|recombinant)', case=False)]
                    drug_primary_gene_counts = count_genes_targeted_kg(drug_targets_primary)
                    ranked_drugs_primary = rank_drugs_by_gene_count_kg(drug_primary_gene_counts)
                    visualize_drug_targets(ranked_drugs_primary, drug_targets_primary, primary_disease)

                elif not genes_primary and genes_comorbidity:
                    st.info(f"No shared genes found. Listing drugs targeting genes associated with {comorbidity}.")
                    drug_targets_comorbidity = get_drugs_targeting_genes_kg(genes_comorbidity, kg_df)
                    drug_targets_comorbidity = drug_targets_comorbidity[~drug_targets_comorbidity['x_name'].str.contains(r'(fibroblast|keratinocyte|neonatal|ovine|recombinant)', case=False)]
                    drug_comorbidity_gene_counts = count_genes_targeted_kg(drug_targets_comorbidity)
                    ranked_drugs_comorbidity = rank_drugs_by_gene_count_kg(drug_comorbidity_gene_counts)
                    visualize_drug_targets(ranked_drugs_comorbidity, drug_targets_comorbidity, comorbidity)

                else:
                    st.warning("No genes found associated with either the primary disease or the comorbidity.")

            else:
                st.warning("Could not find exact or close matches for both diseases in the Knowledge Graph.")

        elif analysis_mode == "LLM + KG":
            st.info("LLM + KG integration will be implemented here.")
            kg_df = load_kg()
            matched_primary_disease = get_closest_disease_name_kg(primary_disease, kg_df)
            matched_comorbidity = get_closest_disease_name_kg(comorbidity, kg_df)

            if not matched_primary_disease or not matched_comorbidity:
                st.warning("Could not find close matches for both diseases in the Knowledge Graph.")
                return

            genes_primary = get_genes_for_disease_kg(matched_primary_disease, kg_df)
            genes_comorbidity = get_genes_for_disease_kg(matched_comorbidity, kg_df)
            shared_genes = list(set(genes_primary) & set(genes_comorbidity))

            drug_targets_shared = get_drugs_targeting_genes_kg(shared_genes, kg_df)
            drug_targets_shared = drug_targets_shared[
                ~drug_targets_shared['x_name'].str.contains(r'(fibroblast|keratinocyte|neonatal|ovine|recombinant)', case=False)
            ]
            drug_shared_gene_counts = count_genes_targeted_kg(drug_targets_shared)
            ranked_drugs_shared = rank_drugs_by_gene_count_kg(drug_shared_gene_counts)

            # Prepare KG context string for prompt
            kg_context = f"Knowledge Graph Context:\nShared Genes ({len(shared_genes)}): {', '.join(shared_genes[:10])}\n"
            kg_context += "Top Drugs Targeting Shared Genes:\n"
            for drug, count in ranked_drugs_shared[:5]:
                targets = drug_targets_shared[drug_targets_shared['x_name'] == drug]['y_name'].unique().tolist()
                kg_context += f"- {drug} targets {count} genes ({', '.join(targets[:3])})\n"

            # Fetch abstracts
            query = build_pubmed_query(primary_disease, comorbidity)
            abstracts = fetch_pubmed_abstracts(query, max_results)

            if not abstracts:
                st.warning("No abstracts found for the diseases.")
                return

            model = configure_gemini()
            full_text = "\n".join([
                f"Title: {record.get('TI', '')}\nAbstract: {record.get('AB', '')}\n"
                for record in abstracts if record.get('AB')
            ])

            prompt = f"""Analyze the following collection of medical abstracts and extract relationships between diseases,
comorbidities, and drugs.

Primary Disease: {primary_disease}
Comorbidity: {comorbidity}

{kg_context}

Your Task:
- List drugs used to treat each disease.
- Identify drugs that are effective for both conditions.
- For each shared drug, provide:
  1. Repurposing score (1–10)(If the drug is not explicitly mentioned in the literature, but is biologically plausible based on the knowledge graph, assign a score between 3–6 based on inferred mechanism.)
  2. Evidence from abstracts
  3. Mechanism of action
  4. Mentioned or likely molecular targets

Output JSON Format:
{{
  "primary_disease": "...",
  "comorbidity": "...",
  "drugs_primary_disease": [...],
  "drugs_comorbidity": [...],
  "shared_treatments": [
    {{
      "drug": "...",
      "repurposing_score": 7,
      "evidence": "...",
      "mechanism_of_action": "...",
      "molecular_targets": [...]
    }}
  ],
  "explanation": "..."
}}

Medical Abstracts:
{full_text}
"""

            try:
                with st.spinner("Analyzing with LLM + KG..."):
                    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                    text_response = re.sub(r'```json\s*', '', response.text)
                    text_response = re.sub(r'```\s*', '', text_response).strip()
                    results = json.loads(text_response)

                st.header(f"Analysis Results (LLM + KG): {primary_disease} + {comorbidity}")
                visualize_drug_rankings(results, approved_drug_names) # Pass drugbank_df
                csv_file = save_to_csv(results, primary_disease, comorbidity)
                if csv_file:
                    with open(csv_file, 'rb') as f:
                        st.download_button("Download CSV", f, file_name=csv_file, mime="text/csv")
                if 'explanation' in results:
                    st.subheader("Explanation")
                    st.write(results['explanation'])
                st.subheader("Raw JSON Output")
                st.json(results)

            except Exception as e:
                st.error(f"Error during LLM + KG analysis: {str(e)}")


if __name__ == "__main__":
    main()
