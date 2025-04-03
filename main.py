#!/usr/bin/env python3
"""
Main script for the Personalized Drug Recommendation System.

This script provides a command-line interface for the system, allowing users
to train models, make predictions, and run simulations.
"""
import os
import argparse

from src.train import train_model
from src.predict import recommend_drugs_for_patient, visualize_recommendations, load_model, load_drug_catalog, process_patient_data
from src.models import VirtualDrugSimulator


def setup_arg_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Personalized Drug Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train drug response prediction model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate drug recommendations for a patient")
    predict_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    predict_parser.add_argument("--config", type=str, required=True, help="Path to model configuration file")
    predict_parser.add_argument("--drug_catalog", type=str, required=True, help="Path to drug catalog file")
    predict_parser.add_argument("--patient_id", type=str, required=True, help="Patient ID")
    predict_parser.add_argument("--mutations", type=str, required=True, help="Path to mutation data file")
    predict_parser.add_argument("--history", type=str, required=True, help="Path to patient history file")
    predict_parser.add_argument("--genetic_model", type=str, required=True, help="Path to genetic model or model name")
    predict_parser.add_argument("--genetic_tokenizer", type=str, required=True, help="Path to genetic tokenizer or tokenizer name")
    predict_parser.add_argument("--top_k", type=int, default=3, help="Number of top drugs to recommend")
    predict_parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    predict_parser.add_argument("--simulate", action="store_true", help="Simulate drug effects")
    predict_parser.add_argument("--genetic_model_checkpoint", type=str, help="Path to genetic model checkpoint for simulation")
    predict_parser.add_argument("--drug_model_checkpoint", type=str, help="Path to drug model checkpoint for simulation")
    
    # Simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Simulate drug effects for a patient")
    simulate_parser.add_argument("--genetic_model_checkpoint", type=str, required=True, help="Path to genetic model checkpoint")
    simulate_parser.add_argument("--drug_model_checkpoint", type=str, required=True, help="Path to drug model checkpoint")
    simulate_parser.add_argument("--patient_id", type=str, required=True, help="Patient ID")
    simulate_parser.add_argument("--drug_id", type=str, required=True, help="Drug ID")
    simulate_parser.add_argument("--mutations", type=str, required=True, help="Path to mutation data file")
    simulate_parser.add_argument("--history", type=str, required=True, help="Path to patient history file")
    simulate_parser.add_argument("--drug_catalog", type=str, required=True, help="Path to drug catalog file")
    simulate_parser.add_argument("--genetic_tokenizer", type=str, required=True, help="Path to genetic tokenizer or tokenizer name")
    simulate_parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    return parser


def process_result_and_exit(success, message, exit_code=1):
    """Process result and exit with appropriate code."""
    if success:
        print(f"✅ {message}")
        exit(0)
    else:
        print(f"❌ {message}")
        exit(exit_code)


def train_command(args):
    """Handle train command."""
    try:
        print(f"Training model with configuration from {args.config}")
        print(f"Config path exists: {os.path.exists(args.config)}")
        train_model(args.config)
        return True, "Training completed successfully."
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return False, f"Error during training: {str(e)}"


def predict_command(args):
    """Handle predict command."""
    try:
        print(f"Generating drug recommendations for patient {args.patient_id}")
        
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
        
        return True, f"Drug recommendations for patient {args.patient_id} saved to {args.output_dir}"
    except Exception as e:
        return False, f"Error during prediction: {str(e)}"


def simulate_command(args):
    """Handle simulate command."""
    try:
        from utils.visualization import plot_efficacy_time_series, plot_side_effects_heatmap
        
        print(f"Simulating drug effects for patient {args.patient_id} and drug {args.drug_id}")
        
        # Load genetic and drug models for simulation
        from src.models import GeneticTransformer, DrugTransformer
        
        genetic_model = GeneticTransformer.load_from_checkpoint(args.genetic_model_checkpoint)
        drug_model = DrugTransformer.load_from_checkpoint(args.drug_model_checkpoint)
        
        # Create simulator
        simulator = VirtualDrugSimulator(
            genetic_model=genetic_model,
            drug_model=drug_model
        )
        
        # Process patient data
        patient_data = process_patient_data(
            args.patient_id,
            args.mutations,
            args.history,
            "dmis-lab/biobert-base-cased-v1.1",  # Default model name
            args.genetic_tokenizer
        )
        
        # Load drug catalog
        drug_catalog = load_drug_catalog(args.drug_catalog)
        
        # Find the specific drug
        drug_data = next((d for d in drug_catalog if d["drug_id"] == args.drug_id), None)
        
        if drug_data is None:
            return False, f"Drug ID {args.drug_id} not found in the catalog"
        
        # Simulate drug effect
        simulation = simulator.simulate_drug_effect(patient_data, drug_data)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Visualize simulation results
        simulations = [simulation]
        
        # Plot efficacy over time
        plot_efficacy_time_series(
            args.patient_id, 
            simulations,
            os.path.join(args.output_dir, f"patient_{args.patient_id}_drug_{args.drug_id}_efficacy.png")
        )
        
        # Plot side effects
        plot_side_effects_heatmap(
            args.patient_id,
            simulations,
            os.path.join(args.output_dir, f"patient_{args.patient_id}_drug_{args.drug_id}_side_effects.png")
        )
        
        return True, f"Drug simulation for patient {args.patient_id} and drug {args.drug_id} saved to {args.output_dir}"
    except Exception as e:
        return False, f"Error during simulation: {str(e)}"


def main():
    """Main function."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        exit(1)
    
    # Handle commands
    if args.command == "train":
        success, message = train_command(args)
    elif args.command == "predict":
        success, message = predict_command(args)
    elif args.command == "simulate":
        success, message = simulate_command(args)
    else:
        success, message = False, f"Unknown command: {args.command}"
    
    process_result_and_exit(success, message)


if __name__ == "__main__":
    main() 