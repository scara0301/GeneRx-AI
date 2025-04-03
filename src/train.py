"""
Training script for the drug response prediction model.
"""
import os
import argparse
import yaml
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, random_split

from src.models import DrugResponsePredictor
from utils.visualization import plot_drug_recommendation_comparison



class DrugResponseDataset(Dataset):
    """Dataset for drug response prediction."""
    
    def __init__(self, data_file: str):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to data file
        """
        self.data = pd.read_csv(data_file)
        
        # Process PharmGKB data format to our model format
        print(f"Loaded dataset with {len(self.data)} rows and columns: {self.data.columns.tolist()}")
        
        # Create input tensors from genetic variant data and drug data
        self.processed_data = []
        
        for _, row in self.data.iterrows():
            # Extract genetic variant info
            variant = str(row['Variant/Haplotypes'])
            gene = str(row['Gene'])
            
            # Extract drug info
            drugs = str(row['Drug(s)'])
            
            # Extract response info (phenotype)
            phenotype = str(row['Phenotype(s)'])
            
            # Create a numeric score from the evidence level
            level_str = str(row['Level of Evidence'])
            # Map evidence levels to numeric values (0 to 4 for 5 classes)
            level_map = {'1A': 4, '1B': 3, '2A': 2, '2B': 1, '3': 0, '4': 0}
            response_label = level_map.get(level_str, 0)
            
            # Generate tokens (simplified version)
            genetic_tokens = self.tokenize_genetic_info(variant, gene)
            drug_tokens = self.tokenize_drug_info(drugs)
            
            # Store processed data
            self.processed_data.append({
                'patient_id': str(row['Clinical Annotation ID']),
                'drug_id': f"drug_{hash(drugs) % 10000:04d}",
                'genetic_input_ids': genetic_tokens['input_ids'],
                'genetic_attention_mask': genetic_tokens['attention_mask'],
                'drug_input_ids': drug_tokens['input_ids'],
                'drug_attention_mask': drug_tokens['attention_mask'],
                'response_label': response_label,
                'phenotype': phenotype
            })
        
        print(f"Processed {len(self.processed_data)} samples")
        
    def tokenize_genetic_info(self, variant, gene):
        """Create simple tokenization for genetic info."""
        # In real implementation, this would use a pretrained tokenizer
        # For simplicity, we just create token IDs based on character codes
        combined = f"{gene}:{variant}"
        # Create input IDs (start with CLS token 101)
        input_ids = [101]
        # Add character codes (limited to 20 chars)
        for char in combined[:20]:
            input_ids.append(ord(char) % 2000 + 1000)  # Keep in reasonable range
        # Add EOS token 102
        input_ids.append(102)
        
        # Create attention mask (1 for all tokens)
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
    def tokenize_drug_info(self, drugs):
        """Create simple tokenization for drug info."""
        # Similar approach as for genetic info
        # For simplicity, we just create token IDs based on character codes
        # Create input IDs (start with CLS token 101)
        input_ids = [101]
        # Add character codes (limited to 20 chars)
        for char in drugs[:20]:
            input_ids.append(ord(char) % 2000 + 3000)  # Different range from genetic tokens
        # Add EOS token 102
        input_ids.append(102)
        
        # Create attention mask (1 for all tokens)
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get dataset item."""
        item = self.processed_data[idx]
        
        return {
            "genetic_input_ids": item["genetic_input_ids"],
            "genetic_attention_mask": item["genetic_attention_mask"],
            "drug_input_ids": item["drug_input_ids"],
            "drug_attention_mask": item["drug_attention_mask"],
            "response_labels": torch.tensor(item["response_label"], dtype=torch.long),
            "patient_id": item["patient_id"],
            "drug_id": item["drug_id"]
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length inputs.
    Pads sequences to the same length within a batch.
    """
    # Get max lengths for padding
    max_genetic_len = max([len(item['genetic_input_ids']) for item in batch])
    max_drug_len = max([len(item['drug_input_ids']) for item in batch])
    
    # Initialize lists to hold batch data
    genetic_input_ids = []
    genetic_attention_mask = []
    drug_input_ids = []
    drug_attention_mask = []
    response_labels = []
    patient_ids = []
    drug_ids = []
    
    # Process each item
    for item in batch:
        # Pad genetic inputs
        g_ids = item['genetic_input_ids']
        g_mask = item['genetic_attention_mask']
        g_padding = max_genetic_len - len(g_ids)
        
        genetic_input_ids.append(torch.cat([g_ids, torch.zeros(g_padding, dtype=torch.long)]))
        genetic_attention_mask.append(torch.cat([g_mask, torch.zeros(g_padding, dtype=torch.long)]))
        
        # Pad drug inputs
        d_ids = item['drug_input_ids']
        d_mask = item['drug_attention_mask']
        d_padding = max_drug_len - len(d_ids)
        
        drug_input_ids.append(torch.cat([d_ids, torch.zeros(d_padding, dtype=torch.long)]))
        drug_attention_mask.append(torch.cat([d_mask, torch.zeros(d_padding, dtype=torch.long)]))
        
        # Add other items
        response_labels.append(item['response_labels'])
        patient_ids.append(item['patient_id'])
        drug_ids.append(item['drug_id'])
    
    # Stack tensors
    return {
        'genetic_input_ids': torch.stack(genetic_input_ids),
        'genetic_attention_mask': torch.stack(genetic_attention_mask),
        'drug_input_ids': torch.stack(drug_input_ids),
        'drug_attention_mask': torch.stack(drug_attention_mask),
        'response_labels': torch.stack(response_labels),
        'patient_id': patient_ids,
        'drug_id': drug_ids
    }


def train_model(config_file: str):
    """
    Train the drug response prediction model.
    
    Args:
        config_file: Path to configuration file
    """
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    pl.seed_everything(config["seed"])
    
    # Load dataset
    data_path = os.path.join(config["data_dir"], config["dataset_file"])
    dataset = DrugResponseDataset(data_path)
    
    # Split dataset
    train_size = int(len(dataset) * config["train_ratio"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
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
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["output_dir"], "checkpoints"),
        filename="drug_response_model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["patience"],
        mode="min"
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(config["output_dir"], "logs"),
        name="drug_response"
    )
    
    # Create trainer
    # Force CPU usage with detailed logging
    print("Using CPU for training")
    print(f"Max epochs: {config['max_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="cpu",
        devices=1,  # Use 1 CPU core
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        log_every_n_steps=config["log_every_n_steps"],
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    trainer.save_checkpoint(os.path.join(config["output_dir"], "drug_response_model_final.ckpt"))
    
    print(f"Training complete. Model saved to {config['output_dir']}/drug_response_model_final.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drug response prediction model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    train_model(args.config) 