"""
Model definitions for drug response prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import pytorch_lightning as pl
from typing import Dict, List


class GeneticTransformer(pl.LightningModule):
    """Transformer model for processing genetic data."""
    
    def __init__(
        self, 
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        num_classes: int = 0,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        dropout_rate: float = 0.1,
        embedding_dim: int = 768,
    ):
        """
        Initialize the genetic transformer model.
        
        Args:
            model_name: Pre-trained model name
            num_classes: Number of classes for classification (0 for feature extraction only)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            dropout_rate: Dropout rate
            embedding_dim: Dimension of the embedding
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Freeze certain layers for fine-tuning
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
            
        # Additional layers
        self.dropout = nn.Dropout(dropout_rate)
        
        if num_classes > 0:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_embeddings: If True, return embeddings instead of logits
            
        Returns:
            Model outputs
        """
        # Pass through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get CLS token embedding
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        if return_embeddings or self.classifier is None:
            return pooled_output
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


class DrugTransformer(pl.LightningModule):
    """Transformer model for processing drug data."""
    
    def __init__(
        self, 
        model_name: str = "Downloads/DrugBERT",
        num_classes: int = 0,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        dropout_rate: float = 0.1,
        embedding_dim: int = 768,
    ):
        """
        Initialize the drug transformer model.
        
        Args:
            model_name: Pre-trained model name or path
            num_classes: Number of classes for classification (0 for feature extraction only)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            dropout_rate: Dropout rate
            embedding_dim: Dimension of the embedding
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Additional layers
        self.dropout = nn.Dropout(dropout_rate)
        
        if num_classes > 0:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_embeddings: If True, return embeddings instead of logits
            
        Returns:
            Model outputs
        """
        # Pass through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get CLS token embedding
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        if return_embeddings or self.classifier is None:
            return pooled_output
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


class DrugResponsePredictor(pl.LightningModule):
    """
    Model to predict drug response for a patient based on genetic data and drug information.
    """
    
    def __init__(
        self,
        genetic_model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        drug_model_name: str = "Downloads/DrugBERT",
        num_response_classes: int = 5,  # e.g., 5 levels of response from no effect to high efficacy
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        dropout_rate: float = 0.2,
        genetic_embedding_dim: int = 768,
        drug_embedding_dim: int = 768,
        fusion_hidden_dim: int = 512,
    ):
        """
        Initialize the drug response predictor.
        
        Args:
            genetic_model_name: Pre-trained genetic model name
            drug_model_name: Pre-trained drug model name
            num_response_classes: Number of response classes
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            dropout_rate: Dropout rate
            genetic_embedding_dim: Dimension of the genetic embedding
            drug_embedding_dim: Dimension of the drug embedding
            fusion_hidden_dim: Dimension of the fusion hidden layer
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Genetic model (feature extractor)
        self.genetic_model = GeneticTransformer(
            model_name=genetic_model_name,
            num_classes=0,  # Use as feature extractor
            embedding_dim=genetic_embedding_dim,
        )
        
        # Drug model (feature extractor)
        self.drug_model = DrugTransformer(
            model_name=drug_model_name,
            num_classes=0,  # Use as feature extractor
            embedding_dim=drug_embedding_dim,
        )
        
        # Fusion layers
        combined_dim = genetic_embedding_dim + drug_embedding_dim
        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Output layer
        self.classifier = nn.Linear(fusion_hidden_dim // 2, num_response_classes)
        
    def forward(
        self, 
        genetic_input_ids, 
        genetic_attention_mask,
        drug_input_ids,
        drug_attention_mask,
    ):
        """
        Forward pass through the model.
        
        Args:
            genetic_input_ids: Genetic input token IDs
            genetic_attention_mask: Genetic attention mask
            drug_input_ids: Drug input token IDs
            drug_attention_mask: Drug attention mask
            
        Returns:
            Response prediction logits
        """
        # Extract genetic features
        genetic_features = self.genetic_model(
            genetic_input_ids, 
            genetic_attention_mask,
            return_embeddings=True
        )
        
        # Extract drug features
        drug_features = self.drug_model(
            drug_input_ids,
            drug_attention_mask,
            return_embeddings=True
        )
        
        # Combine features
        combined_features = torch.cat([genetic_features, drug_features], dim=1)
        
        # Fusion layers
        fused_features = self.fusion_layers(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        genetic_input_ids = batch["genetic_input_ids"]
        genetic_attention_mask = batch["genetic_attention_mask"]
        drug_input_ids = batch["drug_input_ids"]
        drug_attention_mask = batch["drug_attention_mask"]
        labels = batch["response_labels"]
        
        logits = self(
            genetic_input_ids, 
            genetic_attention_mask,
            drug_input_ids,
            drug_attention_mask
        )
        
        loss = F.cross_entropy(logits, labels)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        genetic_input_ids = batch["genetic_input_ids"]
        genetic_attention_mask = batch["genetic_attention_mask"]
        drug_input_ids = batch["drug_input_ids"]
        drug_attention_mask = batch["drug_attention_mask"]
        labels = batch["response_labels"]
        
        logits = self(
            genetic_input_ids, 
            genetic_attention_mask,
            drug_input_ids,
            drug_attention_mask
        )
        
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        # Group parameters for different learning rates
        genetic_params = list(self.genetic_model.parameters())
        drug_params = list(self.drug_model.parameters())
        fusion_params = list(self.fusion_layers.parameters()) + list(self.classifier.parameters())
        
        # Ensure learning rate and weight decay are floats
        learning_rate = float(self.hparams.learning_rate)
        weight_decay = float(self.hparams.weight_decay)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {"params": genetic_params, "lr": learning_rate / 10},  # Lower LR for pre-trained models
            {"params": drug_params, "lr": learning_rate / 10},     # Lower LR for pre-trained models
            {"params": fusion_params, "lr": learning_rate}         # Higher LR for new layers
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


class DrugRecommendationSystem(nn.Module):
    """
    System to recommend drugs based on patient genetic profile and history.
    """
    
    def __init__(
        self,
        drug_response_model: DrugResponsePredictor,
        patient_embedding_dim: int = 768,
        drug_embedding_dim: int = 768,
        top_k: int = 3,
    ):
        """
        Initialize the drug recommendation system.
        
        Args:
            drug_response_model: Pre-trained drug response prediction model
            patient_embedding_dim: Dimension of the patient embedding
            drug_embedding_dim: Dimension of the drug embedding
            top_k: Number of top drugs to recommend
        """
        super().__init__()
        self.drug_response_model = drug_response_model
        self.top_k = top_k
        
    def recommend_drugs(
        self, 
        patient_data: Dict, 
        drug_catalog: List[Dict],
    ) -> List[Dict]:
        """
        Recommend drugs for a patient.
        
        Args:
            patient_data: Patient genetic and history data
            drug_catalog: List of available drugs
            
        Returns:
            List of recommended drugs with predicted responses
        """
        # Extract patient genetic data
        genetic_input_ids = patient_data["genetic_input_ids"]
        genetic_attention_mask = patient_data["genetic_attention_mask"]
        
        # Predict responses for all drugs
        predictions = []
        for drug in drug_catalog:
            drug_input_ids = drug["input_ids"]
            drug_attention_mask = drug["attention_mask"]
            
            # Predict response
            with torch.no_grad():
                logits = self.drug_response_model(
                    genetic_input_ids.unsqueeze(0), 
                    genetic_attention_mask.unsqueeze(0),
                    drug_input_ids.unsqueeze(0),
                    drug_attention_mask.unsqueeze(0),
                )
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=1)
                
                # Calculate weighted score (higher class gets higher weight)
                num_classes = probs.shape[1]
                weights = torch.arange(num_classes, device=probs.device).float()
                weighted_score = torch.sum(probs * weights)
                
                predictions.append({
                    "drug_id": drug["drug_id"],
                    "drug_name": drug["drug_name"],
                    "response_probs": probs.squeeze().tolist(),
                    "score": weighted_score.item()
                })
        
        # Sort by score (descending)
        predictions.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k drugs
        return predictions[:self.top_k]


class VirtualDrugSimulator(nn.Module):
    """
    Simulate drug effects in virtual environments before clinical trials.
    """
    
    def __init__(
        self,
        genetic_model: GeneticTransformer,
        drug_model: DrugTransformer,
        hidden_dim: int = 512,
        num_simulation_steps: int = 10,
    ):
        """
        Initialize the virtual drug simulator.
        
        Args:
            genetic_model: Pre-trained genetic model
            drug_model: Pre-trained drug model
            hidden_dim: Dimension of hidden layers
            num_simulation_steps: Number of steps in the simulation
        """
        super().__init__()
        self.genetic_model = genetic_model
        self.drug_model = drug_model
        self.num_simulation_steps = num_simulation_steps
        
        # Combined dimension
        combined_dim = genetic_model.hparams.embedding_dim + drug_model.hparams.embedding_dim
        
        # Simulation layers
        self.simulation_network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, combined_dim)
        )
        
        # Effect prediction layers
        self.effect_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single value representing efficacy
        )
        
        # Side effect prediction layers
        self.side_effect_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # 5 different types of side effects
        )
    
    def forward(
        self, 
        genetic_input_ids, 
        genetic_attention_mask,
        drug_input_ids,
        drug_attention_mask,
    ):
        """
        Forward pass through the model.
        
        Args:
            genetic_input_ids: Genetic input token IDs
            genetic_attention_mask: Genetic attention mask
            drug_input_ids: Drug input token IDs
            drug_attention_mask: Drug attention mask
            
        Returns:
            Efficacy score and side effects
        """
        # Extract genetic features
        genetic_features = self.genetic_model(
            genetic_input_ids, 
            genetic_attention_mask,
                    return_embeddings=True
        )
        
        # Extract drug features
        drug_features = self.drug_model(
            drug_input_ids,
            drug_attention_mask,
            return_embeddings=True
        )
        
        # Combine features
        combined_features = torch.cat([genetic_features, drug_features], dim=1)
        
        # Run simulation steps
        current_state = combined_features
        all_states = [current_state]
        
        for _ in range(self.num_simulation_steps):
            # Update state based on previous state
            state_update = self.simulation_network(current_state)
            current_state = current_state + 0.1 * state_update  # Small update to simulate time progression
            all_states.append(current_state)
        
        # Predict effects at each time step
        efficacy_scores = []
        side_effects = []
        
        for state in all_states:
            efficacy = self.effect_predictor(state)
            side_effect = self.side_effect_predictor(state)
            
            efficacy_scores.append(efficacy)
            side_effects.append(side_effect)
        
        # Stack time series data
        efficacy_time_series = torch.stack(efficacy_scores, dim=1)
        side_effects_time_series = torch.stack(side_effects, dim=1)
        
        return {
            "efficacy": efficacy_time_series,
            "side_effects": side_effects_time_series,
            "final_efficacy": efficacy_scores[-1],
            "final_side_effects": side_effects[-1]
        }
    
    def simulate_drug_effect(
        self, 
        patient_data: Dict, 
        drug_data: Dict,
    ) -> Dict:
        """
        Simulate drug effect for a specific patient.
        
        Args:
            patient_data: Patient genetic and history data
            drug_data: Drug data
            
        Returns:
            Simulation results
        """
        # Extract data
        genetic_input_ids = patient_data["genetic_input_ids"]
        genetic_attention_mask = patient_data["genetic_attention_mask"]
        drug_input_ids = drug_data["input_ids"]
        drug_attention_mask = drug_data["attention_mask"]
        
        # Run simulation
        with torch.no_grad():
            results = self(
                genetic_input_ids.unsqueeze(0),
                genetic_attention_mask.unsqueeze(0),
                drug_input_ids.unsqueeze(0),
                drug_attention_mask.unsqueeze(0),
            )
            
        # Process results
        efficacy = results["final_efficacy"].item()
        side_effects = F.softmax(results["final_side_effects"], dim=1).squeeze().tolist()
        
        # Create time series for plotting
        efficacy_series = results["efficacy"].squeeze().tolist()
        side_effects_series = F.softmax(results["side_effects"], dim=2).squeeze().tolist()
        
        return {
            "drug_id": drug_data["drug_id"],
            "drug_name": drug_data["drug_name"],
            "efficacy": efficacy,
            "side_effects": side_effects,
            "efficacy_time_series": efficacy_series,
            "side_effects_time_series": side_effects_series,
            "num_steps": self.num_simulation_steps + 1
        } 