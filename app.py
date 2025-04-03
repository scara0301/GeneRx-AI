"""
Streamlit web interface for the Personalized Drug Response Prediction System.
"""
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys
from io import StringIO

# Import the simulation functionality from simulate_drug.py
from simulate_drug import (
    calculate_drug_efficacy,
    simulate_response_over_time,
    get_side_effects,
    INTERACTION_SCORES
)

# Set page configuration
st.set_page_config(
    page_title="Personalized Drug Response",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for clean, minimalistic UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #1E3A8A;
        padding: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results/simulation")

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sample data for the UI
SAMPLE_GENES = ["CYP2C9", "VKORC1", "CYP2C19", "SLCO1B1", "CYP3A4", "CYP2D6", "DPYD", "TPMT"]
SAMPLE_VARIANTS = {
    "CYP2C9": ["*1/*1 (Normal)", "*1/*2 (Intermediate)", "*1/*3 (Intermediate)", "*2/*2 (Poor)", "*2/*3 (Poor)", "*3/*3 (Poor)"],
    "VKORC1": ["-1639G>G (Normal)", "-1639G>A (Reduced)", "-1639A>A (Poor)"],
    "CYP2C19": ["*1/*1 (Normal)", "*1/*2 (Intermediate)", "*1/*17 (Rapid)", "*17/*17 (Ultra-rapid)", "*2/*2 (Poor)"],
    "SLCO1B1": ["521TT (Normal)", "521TC (Intermediate)", "521CC (Poor)"],
    "CYP3A4": ["*1/*1 (Normal)", "*1/*22 (Intermediate)", "*22/*22 (Poor)"],
    "CYP2D6": ["*1/*1 (Normal)", "*1/*4 (Intermediate)", "*4/*4 (Poor)", "*1/*2xN (Ultra-rapid)"],
    "DPYD": ["*1/*1 (Normal)", "*1/*2A (Intermediate)", "*2A/*2A (Poor)"],
    "TPMT": ["*1/*1 (Normal)", "*1/*3 (Intermediate)", "*3/*3 (Poor)"]
}

SAMPLE_DRUGS = [
    {"id": "D001", "name": "Warfarin", "category": "Anticoagulant", "genes": ["CYP2C9", "VKORC1"]},
    {"id": "D002", "name": "Clopidogrel", "category": "Antiplatelet", "genes": ["CYP2C19"]},
    {"id": "D003", "name": "Simvastatin", "category": "Statin", "genes": ["SLCO1B1"]},
    {"id": "D004", "name": "Atorvastatin", "category": "Statin", "genes": ["SLCO1B1", "CYP3A4"]},
    {"id": "D005", "name": "Metoprolol", "category": "Beta Blocker", "genes": ["CYP2D6"]},
    {"id": "D006", "name": "Codeine", "category": "Opioid", "genes": ["CYP2D6"]},
    {"id": "D007", "name": "Fluorouracil", "category": "Chemotherapy", "genes": ["DPYD"]},
    {"id": "D008", "name": "Azathioprine", "category": "Immunosuppressant", "genes": ["TPMT"]}
]

def plot_efficacy(patient_id, drug_name, efficacy_time_series):
    """Create a plot of drug efficacy over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    time_steps = range(len(efficacy_time_series))
    ax.plot(time_steps, efficacy_time_series, 'b-o', linewidth=2.5, markersize=8)
    
    ax.set_title(f"Simulated Response to {drug_name} for Patient {patient_id}", fontsize=14)
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Efficacy Score", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate final value
    final_value = efficacy_time_series[-1]
    ax.annotate(f"Final: {final_value:.2f}", 
                xy=(len(efficacy_time_series)-1, final_value),
                xytext=(len(efficacy_time_series)-2, final_value+0.1),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    fig.tight_layout()
    return fig

def plot_side_effects(patient_id, drug_name, side_effects):
    """Create a plot of side effects as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    effects = list(side_effects.keys())
    probabilities = list(side_effects.values())
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    effects = [effects[i] for i in sorted_indices]
    probabilities = [probabilities[i] for i in sorted_indices]
    
    # Create plot
    bars = ax.barh(effects, probabilities, color='salmon')
    
    ax.set_title(f"Predicted Side Effects of {drug_name}", fontsize=14)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_ylabel("Side Effect", fontsize=12)
    
    # Add percentage labels
    for bar, prob in zip(bars, probabilities):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{prob:.0%}", va='center', fontsize=10)
    
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    return fig

def generate_patient_data(patient_id, variants):
    """Generate patient data dictionary from form inputs."""
    genetic_variants = []
    
    for gene, variant in variants.items():
        if variant:
            # Extract variant ID and function from selected option
            variant_parts = variant.split(' (')
            variant_id = variant_parts[0]
            func = variant_parts[1].replace(')', '')
            
            genetic_variants.append({
                "gene": gene,
                "variant": variant_id,
                "function": func
            })
    
    return {
        "patient_id": patient_id,
        "genetic_variants": genetic_variants,
        "age": st.session_state.age,
        "sex": st.session_state.sex,
        "medical_history": st.session_state.medical_history
    }

def generate_drug_data(selected_drugs):
    """Generate drug data for selected drugs."""
    drug_data = []
    
    for drug_id in selected_drugs:
        # Find drug in sample drugs
        drug = next((d for d in SAMPLE_DRUGS if d["id"] == drug_id), None)
        if drug:
            drug_data.append({
                "drug_id": drug["id"],
                "drug_name": drug["name"],
                "description": f"{drug['category']} medication",
                "target_genes": drug["genes"],
                "mechanism": f"{drug['category']} mechanism"
            })
    
    return drug_data

def run_simulation(patient_data, drugs_data):
    """Run drug response simulation."""
    results = []
    
    for drug in drugs_data:
        # Calculate efficacy
        efficacy = calculate_drug_efficacy(patient_data, drug)
        
        # Simulate response over time
        response_series = simulate_response_over_time(efficacy, steps=12)
        
        # Get side effect profile
        side_effects = get_side_effects(drug["drug_name"])
        
        results.append({
            "drug_id": drug["drug_id"],
            "drug_name": drug["drug_name"],
            "efficacy": efficacy,
            "response_series": response_series,
            "side_effects": side_effects
        })
    
    return results

def save_temp_files(patient_data, drugs_data):
    """Save patient and drug data to temporary files."""
    # Save patient data
    patient_file = DATA_DIR / "temp_patient.json"
    with open(patient_file, 'w') as f:
        json.dump(patient_data, f, indent=2)
    
    # Save drug catalog
    drug_file = DATA_DIR / "temp_drug_catalog.json"
    with open(drug_file, 'w') as f:
        json.dump(drugs_data, f, indent=2)
    
    return patient_file, drug_file

def main():
    # Header
    st.title("💊 Personalized Drug Response Prediction")
    
    # Description
    st.markdown("""
    <div class="info-box">
    This tool predicts how a patient might respond to various medications based on their genetic profile.
    It provides insights on drug efficacy and potential side effects to support personalized treatment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables if they don't exist
    if 'patient_id' not in st.session_state:
        st.session_state.patient_id = "P001"
    if 'age' not in st.session_state:
        st.session_state.age = 65
    if 'sex' not in st.session_state:
        st.session_state.sex = "F"
    if 'medical_history' not in st.session_state:
        st.session_state.medical_history = ["hypertension", "atrial fibrillation"]
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    
    # Sidebar for patient information
    with st.sidebar:
        st.header("Patient Information")
        
        st.session_state.patient_id = st.text_input("Patient ID", value=st.session_state.patient_id)
        st.session_state.age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.age)
        st.session_state.sex = st.selectbox("Sex", options=["M", "F"], index=1 if st.session_state.sex == "F" else 0)
        
        # Medical history as multi-select
        medical_history_options = ["hypertension", "diabetes", "heart disease", "atrial fibrillation", 
                                "asthma", "depression", "anxiety", "COPD", "cancer", "arthritis"]
        st.session_state.medical_history = st.multiselect(
            "Medical History", 
            options=medical_history_options,
            default=st.session_state.medical_history
        )
    
    # Main content area - split into three columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Column 1: Genetic Profile
    with col1:
        st.header("Genetic Profile")
        
        # Dictionary to store selected variants
        variants = {}
        
        # Gene dropdowns
        for gene in SAMPLE_GENES:
            variants[gene] = st.selectbox(
                f"{gene} Variant",
                options=[""] + SAMPLE_VARIANTS[gene],
                index=0,
                key=f"gene_{gene}"
            )
    
    # Column 2: Drug Selection
    with col2:
        st.header("Drug Selection")
        
        # Group drugs by category
        drug_categories = {}
        for drug in SAMPLE_DRUGS:
            if drug["category"] not in drug_categories:
                drug_categories[drug["category"]] = []
            drug_categories[drug["category"]].append(drug)
        
        # Display drugs grouped by category
        selected_drugs = []
        for category, drugs in drug_categories.items():
            st.subheader(category)
            for drug in drugs:
                if st.checkbox(f"{drug['name']} ({', '.join(drug['genes'])})", key=f"drug_{drug['id']}"):
                    selected_drugs.append(drug["id"])
        
        # Button to run simulation
        if st.button("Run Simulation", key="run_simulation"):
            if not selected_drugs:
                st.error("Please select at least one drug to simulate")
            else:
                with st.spinner("Running simulation..."):
                    # Generate patient data
                    patient_data = generate_patient_data(st.session_state.patient_id, variants)
                    
                    # Generate drug data
                    drugs_data = generate_drug_data(selected_drugs)
                    
                    # Run simulation
                    st.session_state.simulation_results = run_simulation(patient_data, drugs_data)
    
    # Column 3: Results
    with col3:
        st.header("Simulation Results")
        
        if st.session_state.simulation_results:
            # Tab view for multiple drugs
            tabs = st.tabs([result["drug_name"] for result in st.session_state.simulation_results])
            
            # Display results for each drug
            for i, tab in enumerate(tabs):
                with tab:
                    result = st.session_state.simulation_results[i]
                    
                    # Efficacy score
                    st.metric("Predicted Efficacy", f"{result['efficacy']:.2f}")
                    
                    # Efficacy plot
                    st.pyplot(plot_efficacy(
                        st.session_state.patient_id,
                        result["drug_name"],
                        result["response_series"]
                    ))
                    
                    # Side effects plot
                    st.pyplot(plot_side_effects(
                        st.session_state.patient_id,
                        result["drug_name"],
                        result["side_effects"]
                    ))
        else:
            st.info("Select patient genetic profile and drugs, then run the simulation to see results")
    
    # Advanced options at the bottom
    with st.expander("Advanced Options"):
        st.subheader("Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # File upload for custom patient data
            st.file_uploader("Upload Patient Data CSV", type=["csv"], key="patient_data_upload")
            
        with col2:
            # Training options
            st.number_input("Training Epochs", min_value=1, max_value=100, value=30, key="training_epochs")
            st.selectbox("Genetic Model", options=["BioBERT", "Custom"], index=0, key="genetic_model")
            
        # Train model button
        if st.button("Train Custom Model", key="train_model"):
            st.info("This would start training a model using the provided dataset. Not implemented in this demo.")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2023 Personalized Drug AI | [Documentation](https://github.com/yourusername/personalized_drug_ai)")

if __name__ == "__main__":
    main() 