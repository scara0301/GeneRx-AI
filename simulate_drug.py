"""
Simplified drug effect simulation script.
This version simulates a drug's effect without needing a full trained model.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define drug-gene interaction matrix (simplified)
INTERACTION_SCORES = {
    "CYP2C9": {
        "warfarin": 0.9,     # Strong interaction
        "clopidogrel": 0.2,   # Weak interaction
        "simvastatin": 0.1    # Minimal interaction
    },
    "VKORC1": {
        "warfarin": 0.95,    # Very strong interaction
        "clopidogrel": 0.1,   # Minimal interaction
        "simvastatin": 0.05   # Minimal interaction
    },
    "CYP2C19": {
        "warfarin": 0.3,     # Moderate interaction
        "clopidogrel": 0.85,  # Strong interaction
        "simvastatin": 0.15   # Weak interaction
    },
    "SLCO1B1": {
        "warfarin": 0.15,    # Weak interaction
        "clopidogrel": 0.2,   # Weak interaction
        "simvastatin": 0.8    # Strong interaction
    }
}

# Side effect profiles (simplified)
SIDE_EFFECT_PROFILES = {
    "warfarin": {
        "Bleeding": 0.4,
        "Bruising": 0.35,
        "Headache": 0.15,
        "Nausea": 0.1,
        "Rash": 0.05
    },
    "clopidogrel": {
        "Bleeding": 0.3,
        "Bruising": 0.25,
        "Headache": 0.2,
        "Nausea": 0.15,
        "Rash": 0.1
    },
    "simvastatin": {
        "Muscle Pain": 0.3,
        "Headache": 0.2,
        "Nausea": 0.15,
        "Fatigue": 0.25,
        "Rash": 0.1
    }
}

def load_patient(file_path):
    """Load patient data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_drug_catalog(file_path):
    """Load drug catalog from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_drug_efficacy(patient, drug):
    """Calculate expected drug efficacy based on genetic variants."""
    efficacy_score = 0.5  # Base efficacy
    weight = 0
    
    # Extract genes from patient variants
    patient_genes = [variant["gene"] for variant in patient["genetic_variants"]]
    
    # Check for gene-drug interactions
    for gene in patient_genes:
        if gene in INTERACTION_SCORES and drug["drug_name"].lower() in INTERACTION_SCORES[gene]:
            interaction_score = INTERACTION_SCORES[gene][drug["drug_name"].lower()]
            
            # Find the variant with matching gene
            for variant in patient["genetic_variants"]:
                if variant["gene"] == gene:
                    # Poor metabolizers reduce efficacy for drugs metabolized by that gene
                    if "poor metabolizer" in variant["function"].lower():
                        if interaction_score > 0.5:  # Strong interaction
                            efficacy_score -= 0.2
                        weight += 1
                    # Reduced function also affects efficacy
                    elif "reduced function" in variant["function"].lower():
                        if interaction_score > 0.5:  # Strong interaction
                            efficacy_score -= 0.1
                        weight += 0.5
    
    # Ensure efficacy is within [0, 1]
    return max(0.1, min(0.9, efficacy_score))

def simulate_response_over_time(efficacy, steps=10):
    """Simulate drug response over time."""
    # Start with 0 efficacy and gradually approach target efficacy
    time_series = [0]
    
    for i in range(1, steps):
        # Simulate approach to target efficacy with some noise
        progress = i / steps
        current = time_series[-1]
        target_diff = efficacy - current
        step = target_diff * progress * 0.5  # Gradually approach target
        noise = np.random.normal(0, 0.03)  # Small random fluctuations
        
        new_value = current + step + noise
        # Ensure values stay within reasonable bounds
        new_value = max(0, min(1, new_value))
        time_series.append(new_value)
    
    # Final value should be close to calculated efficacy
    time_series.append(efficacy)
    return time_series

def get_side_effects(drug_name):
    """Get side effect profile for a drug."""
    drug_name = drug_name.lower()
    if drug_name in SIDE_EFFECT_PROFILES:
        return SIDE_EFFECT_PROFILES[drug_name]
    
    # Default side effect profile
    return {
        "Headache": 0.2,
        "Nausea": 0.15,
        "Rash": 0.1,
        "Fatigue": 0.15,
        "Dizziness": 0.1
    }

def plot_efficacy(patient_id, drug_name, efficacy_time_series, output_path):
    """Plot drug efficacy over time."""
    plt.figure(figsize=(10, 6))
    
    time_steps = range(len(efficacy_time_series))
    plt.plot(time_steps, efficacy_time_series, 'b-o', linewidth=2.5, markersize=8)
    
    plt.title(f"Simulated Response to {drug_name} for Patient {patient_id}", fontsize=16)
    plt.xlabel("Time (days)", fontsize=14)
    plt.ylabel("Efficacy Score", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate final value
    final_value = efficacy_time_series[-1]
    plt.annotate(f"Final: {final_value:.2f}", 
                xy=(len(efficacy_time_series)-1, final_value),
                xytext=(len(efficacy_time_series)-2, final_value+0.1),
                fontsize=12,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Efficacy plot saved to {output_path}")

def plot_side_effects(patient_id, drug_name, side_effects, output_path):
    """Plot side effects as a bar chart."""
    plt.figure(figsize=(10, 6))
    
    effects = list(side_effects.keys())
    probabilities = list(side_effects.values())
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    effects = [effects[i] for i in sorted_indices]
    probabilities = [probabilities[i] for i in sorted_indices]
    
    # Create plot
    bars = plt.barh(effects, probabilities, color='salmon')
    
    plt.title(f"Predicted Side Effects of {drug_name} for Patient {patient_id}", fontsize=16)
    plt.xlabel("Probability", fontsize=14)
    plt.ylabel("Side Effect", fontsize=14)
    
    # Add percentage labels
    for bar, prob in zip(bars, probabilities):
        plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{prob:.0%}", va='center', fontsize=12)
    
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Side effects plot saved to {output_path}")

def main():
    """Main simulation function."""
    # Paths
    data_dir = Path("personalized_drug_ai/data")
    results_dir = Path("personalized_drug_ai/results/simulation")
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    patient = load_patient(data_dir / "test_patient.json")
    catalog = load_drug_catalog(data_dir / "drug_catalog.json")
    
    # Process each drug
    for drug in catalog:
        print(f"\nSimulating response to {drug['drug_name']} for patient {patient['patient_id']}...")
        
        # Calculate efficacy
        efficacy = calculate_drug_efficacy(patient, drug)
        print(f"Predicted efficacy: {efficacy:.2f}")
        
        # Simulate response over time
        response_series = simulate_response_over_time(efficacy, steps=12)
        print(f"Response over time: {[round(v, 2) for v in response_series]}")
        
        # Get side effect profile
        side_effects = get_side_effects(drug['drug_name'])
        
        # Plot results
        plot_efficacy(
            patient['patient_id'],
            drug['drug_name'],
            response_series,
            results_dir / f"patient_{patient['patient_id']}_drug_{drug['drug_id']}_efficacy.png"
        )
        
        plot_side_effects(
            patient['patient_id'],
            drug['drug_name'],
            side_effects,
            results_dir / f"patient_{patient['patient_id']}_drug_{drug['drug_id']}_side_effects.png"
        )
    
    print("\nSimulation complete. Results saved to:", results_dir)


if __name__ == "__main__":
    main() 