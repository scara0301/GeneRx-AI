"""
Prediction script for drug recommendations.
"""
import os
import argparse
import yaml
import json
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import DrugResponsePredictor, DrugRecommendationSystem, VirtualDrugSimulator, GeneticTransformer, DrugTransformer
from src.data_processing import GeneticDataProcessor, DrugDataProcessor, PatientDataProcessor

def load_model(checkpoint_path: str, config_file: str) -> DrugResponsePredictor:
    """
    Load a trained drug response prediction model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_file: Path to model configuration file
        
    Returns:
        Loaded model
    """
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Create model with same configuration
    model = DrugResponsePredictor(
        genetic_model_name=config["genetic_model_name"],
        drug_model_name=config["drug_model_name"],
        num_response_classes=config["num_response_classes"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        dropout_rate=config["dropout_rate"],
        genetic_embedding_dim=config["genetic_embedding_dim"],
        drug_embedding_dim=config["drug_embedding_dim"],
        fusion_hidden_dim=config["fusion_hidden_dim"]
    )
    
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    model.eval()
    
    return model


def load_drug_catalog(drug_catalog_file: str) -> List[Dict]:
    """
    Load drug catalog from file.
    
    Args:
        drug_catalog_file: Path to drug catalog file
        
    Returns:
        List of drug information dictionaries
    """
    drugs = pd.read_csv(drug_catalog_file)
    
    # Convert string tensors to actual tensors
    for column in ["input_ids", "attention_mask"]:
        if column in drugs.columns:
            drugs[column] = drugs[column].apply(lambda x: torch.tensor(eval(x)))
    
    drug_catalog = []
    for _, row in drugs.iterrows():
        drug_catalog.append({
            "drug_id": row["drug_id"],
            "drug_name": row["drug_name"],
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "smiles": row["smiles"] if "smiles" in row else None,
            "description": row["description"] if "description" in row else None
        })
    
    return drug_catalog


def process_patient_data(
    patient_id: str,
    mutation_file: str,
    history_file: str,
    genetic_model_path: str,
    genetic_tokenizer_path: str,
) -> Dict:
    """
    Process patient data for prediction.
    
    Args:
        patient_id: Patient ID
        mutation_file: Path to mutation data file
        history_file: Path to patient history file
        genetic_model_path: Path to genetic model or name
        genetic_tokenizer_path: Path to genetic tokenizer or name
        
    Returns:
        Processed patient data
    """
    # Initialize processors
    genetic_processor = GeneticDataProcessor(tokenizer_name=genetic_tokenizer_path)
    patient_processor = PatientDataProcessor(tokenizer_name=genetic_tokenizer_path)
    
    # Load mutation data
    mutations_df = genetic_processor.process_mutation_data(mutation_file)
    mutations_df = mutations_df[mutations_df["patient_id"] == patient_id]
    
    if len(mutations_df) == 0:
        raise ValueError(f"Patient ID {patient_id} not found in mutation data")
    
    # Load patient history
    history_df = patient_processor.process_patient_history(history_file)
    history_df = history_df[history_df["patient_id"] == patient_id]
    
    if len(history_df) == 0:
        raise ValueError(f"Patient ID {patient_id} not found in patient history data")
    
    # Get first mutation (simplified for demo - in practice would combine multiple mutations)
    mutation_row = mutations_df.iloc[0]
    history_row = history_df.iloc[0]
    
    # Create patient data dictionary
    patient_data = {
        "patient_id": patient_id,
        "genetic_input_ids": mutation_row["input_ids"],
        "genetic_attention_mask": mutation_row["attention_mask"],
        "history_input_ids": history_row["input_ids"],
        "history_attention_mask": history_row["attention_mask"],
        "mutation_desc": mutation_row["mutation_desc"],
        "history_text": history_row["history_text"]
    }
    
    return patient_data


def recommend_drugs_for_patient(
    patient_data: Dict,
    drug_catalog: List[Dict],
    model: DrugResponsePredictor,
    top_k: int = 3,
    simulate: bool = False,
    genetic_model_path: str = None,
    drug_model_path: str = None,
) -> Dict:
    """
    Recommend drugs for a patient.
    
    Args:
        patient_data: Processed patient data
        drug_catalog: Drug catalog
        model: Trained drug response prediction model
        top_k: Number of top drugs to recommend
        simulate: Whether to simulate drug effects
        genetic_model_path: Path to genetic model for simulation
        drug_model_path: Path to drug model for simulation
        
    Returns:
        Recommendation results
    """
    # Create recommendation system
    recommendation_system = DrugRecommendationSystem(
        drug_response_model=model,
        top_k=top_k
    )
    
    # Get recommendations
    recommendations = recommendation_system.recommend_drugs(patient_data, drug_catalog)
    
    results = {
        "patient_id": patient_data["patient_id"],
        "recommendations": recommendations
    }
    
    # Simulate drug effects if requested
    if simulate:
        if not genetic_model_path or not drug_model_path:
            raise ValueError("Genetic and drug model paths must be provided for simulation")
        
        # Load genetic and drug models for simulation
        genetic_model = GeneticTransformer.load_from_checkpoint(genetic_model_path)
        drug_model = DrugTransformer.load_from_checkpoint(drug_model_path)
        
        # Create simulator
        simulator = VirtualDrugSimulator(
            genetic_model=genetic_model,
            drug_model=drug_model
        )
        
        # Simulate effects for recommended drugs
        simulation_results = []
        for drug in recommendations:
            drug_data = next(d for d in drug_catalog if d["drug_id"] == drug["drug_id"])
            simulation = simulator.simulate_drug_effect(patient_data, drug_data)
            simulation_results.append(simulation)
        
        results["simulations"] = simulation_results
    
    return results


def visualize_recommendations(results: Dict, output_dir: str):
    """
    Visualize drug recommendations.
    
    Args:
        results: Recommendation results
        output_dir: Output directory for visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    drug_names = [rec["drug_name"] for rec in results["recommendations"]]
    scores = [rec["score"] for rec in results["recommendations"]]
    
    # Create bar chart of drug scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x=drug_names, y=scores)
    plt.title(f"Drug Recommendations for Patient {results['patient_id']}")
    plt.xlabel("Drug")
    plt.ylabel("Recommendation Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"patient_{results['patient_id']}_recommendations.png"))
    plt.close()
    
    # Visualize response probabilities
    plt.figure(figsize=(12, 8))
    for i, rec in enumerate(results["recommendations"]):
        plt.subplot(1, len(results["recommendations"]), i+1)
        sns.barplot(x=list(range(len(rec["response_probs"]))), y=rec["response_probs"])
        plt.title(rec["drug_name"])
        plt.xlabel("Response Class")
        plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"patient_{results['patient_id']}_response_probs.png"))
    plt.close()
    
    # Visualize simulations if available
    if "simulations" in results:
        # Efficacy over time
        plt.figure(figsize=(12, 6))
        for sim in results["simulations"]:
            plt.plot(range(sim["num_steps"]), sim["efficacy_time_series"], label=sim["drug_name"])
        plt.title(f"Predicted Efficacy Over Time for Patient {results['patient_id']}")
        plt.xlabel("Simulation Step")
        plt.ylabel("Efficacy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"patient_{results['patient_id']}_efficacy.png"))
        plt.close()
        
        # Side effects over time
        plt.figure(figsize=(14, 10))
        side_effect_types = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5"]
        for i, sim in enumerate(results["simulations"]):
            plt.subplot(len(results["simulations"]), 1, i+1)
            side_effects_data = np.array(sim["side_effects_time_series"])
            for j, effect_type in enumerate(side_effect_types):
                plt.plot(range(sim["num_steps"]), side_effects_data[:, j], label=effect_type)
            plt.title(f"{sim['drug_name']} Side Effects")
            plt.xlabel("Simulation Step")
            plt.ylabel("Probability")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"patient_{results['patient_id']}_side_effects.png"))
        plt.close()


def main(args):
    """Main function."""
    # Load model
    model = load_model(args.checkpoint, args.config)
    
    # Load drug catalog
    drug_catalog = load_drug_catalog(args.drug_catalog)
    
    # Process patient data
    patient_data = process_patient_data(
        args.patient_id,
        args.mutations,
        args.history,
        args.genetic_model,
        args.genetic_tokenizer
    )
    
    # Get recommendations
    results = recommend_drugs_for_patient(
        patient_data,
        drug_catalog,
        model,
        args.top_k,
        args.simulate,
        args.genetic_model_checkpoint,
        args.drug_model_checkpoint
    )
    
    # Visualize recommendations
    visualize_recommendations(results, args.output_dir)
    
    # Save results
    with open(os.path.join(args.output_dir, f"patient_{args.patient_id}_recommendations.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Recommendations for patient {args.patient_id} saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate drug recommendations for a patient")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model configuration file")
    parser.add_argument("--drug_catalog", type=str, required=True, help="Path to drug catalog file")
    parser.add_argument("--patient_id", type=str, required=True, help="Patient ID")
    parser.add_argument("--mutations", type=str, required=True, help="Path to mutation data file")
    parser.add_argument("--history", type=str, required=True, help="Path to patient history file")
    parser.add_argument("--genetic_model", type=str, required=True, help="Path to genetic model or model name")
    parser.add_argument("--genetic_tokenizer", type=str, required=True, help="Path to genetic tokenizer or tokenizer name")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top drugs to recommend")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--simulate", action="store_true", help="Simulate drug effects")
    parser.add_argument("--genetic_model_checkpoint", type=str, help="Path to genetic model checkpoint for simulation")
    parser.add_argument("--drug_model_checkpoint", type=str, help="Path to drug model checkpoint for simulation")
    
    args = parser.parse_args()
    
    main(args) 