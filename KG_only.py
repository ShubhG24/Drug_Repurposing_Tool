import pandas as pd
import difflib
from collections import defaultdict

# Load and normalize KG
kg_df = pd.read_csv("kg.csv", low_memory=False)
kg_df['x_type'] = kg_df['x_type'].str.lower()
kg_df['y_type'] = kg_df['y_type'].str.lower()
kg_df['relation'] = kg_df['relation'].str.lower()
kg_df['display_relation'] = kg_df['display_relation'].str.lower()

# Define valid relation sets
GENE_RELEVANT_RELATIONS = {'associated with', 'expression present', 'expression absent'}
DRUG_RELEVANT_RELATIONS = {'target', 'enzyme', 'carrier', 'transporter', 'ppi'}

# ---------------- Fuzzy disease match ----------------
def get_closest_disease_name(disease_name, kg_df):
    disease_name_lower = disease_name.lower()
    disease_names = kg_df[kg_df['x_type'] == 'disease']['x_name'].dropna().str.lower().unique()

    #Exact match
    if disease_name_lower in disease_names:
        return disease_name_lower
    #Fuzzy match
    match = difflib.get_close_matches(disease_name_lower, disease_names, n=1, cutoff=0.6)
    return match[0] if match else None

# ---------------- Gene retrieval ----------------
def get_genes_for_disease(disease_name, kg_df):
    matched_name = get_closest_disease_name(disease_name, kg_df)
    if not matched_name:
        print(f"No close match found for disease: {disease_name}")
        return []
    print(f"Using matched disease name: {matched_name}")
    return kg_df[
        (kg_df['x_name'].str.lower() == matched_name) &
        (kg_df['x_type'] == 'disease') &
        (kg_df['y_type'] == 'gene/protein') &
        (kg_df['display_relation'].isin(GENE_RELEVANT_RELATIONS))
    ]['y_name'].dropna().unique().tolist()

# ---------------- Drug retrieval targeting shared genes ----------------
def get_drugs_targeting_shared_genes(shared_genes, kg_df):
    return kg_df[
        (kg_df['y_name'].isin(shared_genes)) &
        (kg_df['x_type'] == 'drug') &
        (kg_df['y_type'] == 'gene/protein') &
        (kg_df['display_relation'].isin(DRUG_RELEVANT_RELATIONS))
    ][['x_name', 'y_name']].drop_duplicates()

# ---------------- Count shared genes targeted by each drug ----------------
def count_shared_genes_targeted(drug_targets):
    drug_gene_counts = defaultdict(set)
    for _, row in drug_targets.iterrows():
        drug = row['x_name']
        gene = row['y_name']
        drug_gene_counts[drug].add(gene)
    return {drug: len(genes) for drug, genes in drug_gene_counts.items()}

# ---------------- Run pipeline ----------------

primary_disease = "cirrhosis of liver"
comorbidity = "chronic kidney disease"

genes_primary = get_genes_for_disease(primary_disease, kg_df)
genes_comorbidity = get_genes_for_disease(comorbidity, kg_df)
shared_genes = list(set(genes_primary) & set(genes_comorbidity))

print("\nGenes/Proteins for Primary:", genes_primary)
print("\nGenes/Proteins for Comorbidity:", genes_comorbidity)
print("\nShared Genes/Proteins:", shared_genes)

# Get all drug-target interactions involving shared genes
drug_targets_shared = get_drugs_targeting_shared_genes(shared_genes, kg_df)
drug_targets_shared = drug_targets_shared[~drug_targets_shared['x_name'].str.contains(r'(fibroblast|keratinocyte|neonatal|ovine|recombinant)', case=False)]

# Count how many unique shared genes each drug targets
drug_shared_gene_counts = count_shared_genes_targeted(drug_targets_shared)

# Rank drugs by the number of shared genes targeted
ranked_drugs_by_shared_gene_count = sorted(drug_shared_gene_counts.items(), key=lambda item: item[1], reverse=True)

print("\nTop Repurposed Drug Candidates (Ranked by Number of Shared Genes Targeted):")
for drug, count in ranked_drugs_by_shared_gene_count[:10]:
    # Get the specific shared genes targeted by this drug for better explanation
    targeted_shared_genes = drug_targets_shared[drug_targets_shared['x_name'] == drug]['y_name'].unique().tolist()
    print(f"\n{drug}: Targets {count} shared genes - {targeted_shared_genes}")