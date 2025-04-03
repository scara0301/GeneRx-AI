"""
Data processing module for genetic profiles and drug data.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from Bio import SeqIO
from transformers import AutoTokenizer


class GeneticDataProcessor:
    """Process genetic data for model input."""
    
    def __init__(self, tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        """
        Initialize the genetic data processor.
        
        Args:
            tokenizer_name: Name of the pre-trained tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def process_mutation_data(self, mutation_file: str) -> pd.DataFrame:
        """
        Process genetic mutation data.
        
        Args:
            mutation_file: Path to mutation data file
            
        Returns:
            Processed mutation data as DataFrame
        """
        # Load mutation data
        mutations_df = pd.read_csv(mutation_file)
        
        # Process mutations
        processed_data = []
        for _, row in mutations_df.iterrows():
            mutation_desc = f"{row['gene']} {row['mutation_type']} {row['position']} {row['reference']} to {row['variant']}"
            tokenized = self.tokenizer(mutation_desc, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            
            processed_data.append({
                "patient_id": row["patient_id"],
                "mutation_desc": mutation_desc,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            })
            
        return pd.DataFrame(processed_data)
    
    def process_protein_sequence(self, sequence_file: str) -> Dict[str, Dict]:
        """
        Process protein sequence data.
        
        Args:
            sequence_file: Path to protein sequence file (FASTA format)
            
        Returns:
            Dict with protein IDs and their processed sequences
        """
        sequences = {}
        
        for record in SeqIO.parse(sequence_file, "fasta"):
            seq_str = str(record.seq)
            seq_chunks = [seq_str[i:i+512] for i in range(0, len(seq_str), 512)]
            
            processed_chunks = []
            for chunk in seq_chunks:
                tokenized = self.tokenizer(chunk, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                processed_chunks.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                })
            
            sequences[record.id] = {
                "full_sequence": seq_str,
                "processed_chunks": processed_chunks
            }
            
        return sequences


class DrugDataProcessor:
    """Process drug data for model input."""
    
    def __init__(self, tokenizer_name: str = "Downloads/DrugBERT"):
        """
        Initialize the drug data processor.
        
        Args:
            tokenizer_name: Name or path of the pre-trained tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def process_drug_smiles(self, smiles_file: str) -> pd.DataFrame:
        """
        Process drug SMILES notations.
        
        Args:
            smiles_file: Path to drug SMILES data file
            
        Returns:
            Processed drug data as DataFrame
        """
        # Load drug data
        drugs_df = pd.read_csv(smiles_file)
        
        # Process SMILES
        processed_data = []
        for _, row in drugs_df.iterrows():
            tokenized = self.tokenizer(row["smiles"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
            
            processed_data.append({
                "drug_id": row["drug_id"],
                "drug_name": row["drug_name"],
                "smiles": row["smiles"],
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            })
            
        return pd.DataFrame(processed_data)
    
    def process_drug_descriptions(self, descriptions_file: str) -> pd.DataFrame:
        """
        Process drug textual descriptions.
        
        Args:
            descriptions_file: Path to drug descriptions data file
            
        Returns:
            Processed drug descriptions as DataFrame
        """
        # Load drug descriptions
        descriptions_df = pd.read_csv(descriptions_file)
        
        # Process descriptions
        processed_data = []
        for _, row in descriptions_df.iterrows():
            tokenized = self.tokenizer(
                row["description"], 
                padding="max_length", 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
            
            processed_data.append({
                "drug_id": row["drug_id"],
                "drug_name": row["drug_name"],
                "description": row["description"],
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            })
            
        return pd.DataFrame(processed_data)


class PatientDataProcessor:
    """Process patient data for model input."""
    
    def __init__(self, tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        """
        Initialize the patient data processor.
        
        Args:
            tokenizer_name: Name of the pre-trained tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def process_patient_history(self, history_file: str) -> pd.DataFrame:
        """
        Process patient medical history.
        
        Args:
            history_file: Path to patient history data file
            
        Returns:
            Processed patient history as DataFrame
        """
        # Load patient history
        history_df = pd.read_csv(history_file)
        
        # Process history
        processed_data = []
        for _, row in history_df.iterrows():
            history_text = f"Age: {row['age']}. Sex: {row['sex']}. "
            
            if "medical_conditions" in row and not pd.isna(row["medical_conditions"]):
                history_text += f"Conditions: {row['medical_conditions']}. "
                
            if "allergies" in row and not pd.isna(row["allergies"]):
                history_text += f"Allergies: {row['allergies']}. "
                
            if "previous_medications" in row and not pd.isna(row["previous_medications"]):
                history_text += f"Previous medications: {row['previous_medications']}. "
            
            tokenized = self.tokenizer(
                history_text, 
                padding="max_length", 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            )
            
            processed_data.append({
                "patient_id": row["patient_id"],
                "history_text": history_text,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            })
            
        return pd.DataFrame(processed_data)


def merge_patient_data(genetic_data: pd.DataFrame, history_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge genetic and patient history data.
    
    Args:
        genetic_data: Processed genetic data
        history_data: Processed patient history data
        
    Returns:
        Merged patient data
    """
    # Merge on patient_id
    merged_data = genetic_data.merge(history_data, on="patient_id", how="inner")
    return merged_data


if __name__ == "__main__":
    # Example usage
    genetic_processor = GeneticDataProcessor()
    drug_processor = DrugDataProcessor()
    patient_processor = PatientDataProcessor()
    
    # Process sample data if files exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    sample_files = {
        "mutations": os.path.join(data_dir, "sample_mutations.csv"),
        "proteins": os.path.join(data_dir, "sample_proteins.fasta"),
        "drugs": os.path.join(data_dir, "sample_drugs.csv"),
        "drug_desc": os.path.join(data_dir, "sample_drug_descriptions.csv"),
        "patient_history": os.path.join(data_dir, "sample_patient_history.csv"),
    }
    
    for name, file_path in sample_files.items():
        if os.path.exists(file_path):
            print(f"Processing {name} data from {file_path}")
            
            if name == "mutations":
                genetic_processor.process_mutation_data(file_path)
            elif name == "proteins":
                genetic_processor.process_protein_sequence(file_path)
            elif name == "drugs":
                drug_processor.process_drug_smiles(file_path)
            elif name == "drug_desc":
                drug_processor.process_drug_descriptions(file_path)
            elif name == "patient_history":
                patient_processor.process_patient_history(file_path)
                
            print(f"Processed {name} data successfully")
        else:
            print(f"Sample file {file_path} does not exist") 