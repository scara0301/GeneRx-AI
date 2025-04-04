# Personalized Drug Recommendation System

An AI-powered system that predicts personalized drug responses based on genetic profiles and recommends optimal medications.

## Features

- **Genetic Data Processing**: Process raw genetic data to extract relevant variants for drug metabolism
- **Drug Data Integration**: Integrate pharmaceutical data with genetic information
- **ML-based Prediction**: Predict drug efficacy and side effects for individual patients
- **Visualization**: Visualize predicted drug responses and comparisons
- **User Interface**: Interactive web interface for exploring drug responses

## Project Structure

```
personalized_drug_ai/
├── data/                 # Training and test datasets
├── models/               # Trained models
├── results/              # Simulation results and visualizations
├── src/                  # Source code
│   ├── data_processing.py  # Data preprocessing modules
│   ├── models.py           # Model architecture definitions
│   ├── train.py            # Training scripts
│   └── predict.py          # Prediction and inference
├── utils/                # Utility functions
│   └── visualization.py    # Visualization tools
├── configs/              # Configuration files
├── app.py                # Streamlit web application
├── simulate_drug.py      # Drug response simulation script
└── requirements.txt      # Dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### GPU Support (Optional)

For GPU acceleration, you'll need:

1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 11.8 or 12.x
   - Download from: https://developer.nvidia.com/cuda-downloads
3. Install PyTorch with CUDA support:
   ```
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

**Note**: PyTorch with CUDA support for Python 3.13 may not be available yet. If using Python 3.13, you may need to use CPU mode or downgrade to Python 3.10-3.12.

## Usage

### Running the Web Interface

The system includes a user-friendly Streamlit interface for interactive drug response prediction:

1. Ensure all dependencies are installed:
   ```
   pip install streamlit
   ```

2. Run the Streamlit app:
   ```
   cd personalized_drug_ai
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501` to use the interface.

The UI allows you to:
- Input patient genetic variants and medical information
- Select drugs to simulate
- View visualizations of predicted efficacy and side effects
- Compare multiple drug responses

### Adding Data to the Model

To train the model with your own genetic and drug data:

1. Prepare your training data:
   - Place genetic data files in `data/genetic/`
   - Place drug data files in `data/drugs/`
   - Format data according to PharmGKB standards or use custom format as in the examples

2. Create a config file in the `configs/` directory:
   ```json
   {
     "genetic_data_path": "data/genetic/my_data.csv",
     "drug_data_path": "data/drugs/my_drugs.json",
     "model_params": {
       "embedding_dim": 128,
       "hidden_dim": 256,
       "num_layers": 2,
       "dropout": 0.1
     },
     "training_params": {
       "epochs": 30,
       "batch_size": 32,
       "learning_rate": 3e-4
     },
     "output_dir": "models/my_model"
   }
   ```

3. Train the model:
   ```
   python src/train.py --config configs/my_config.json
   ```

### Drug Response Simulation

There are two ways to simulate drug responses:

#### 1. Quick Simulation (No trained model required)

You can run a simplified simulation using pharmacogenomic rules:

```python
python simulate_drug.py
```

This uses predefined gene-drug interaction data to simulate responses.

#### 2. Model-Based Simulation (Trained model required)

For more accurate predictions using the trained model:

```python
python src/predict.py --patient data/patients/P001.json --drugs data/drugs/catalog.json
```

### Data Format Examples

#### Patient Data (JSON)
```json
{
  "patient_id": "P001",
  "genetic_variants": [
    {"gene": "CYP2C9", "variant": "*3/*3", "function": "Poor metabolizer"},
    {"gene": "VKORC1", "variant": "-1639G>A", "function": "Reduced function"}
  ],
  "age": 65,
  "sex": "F",
  "medical_history": ["hypertension", "atrial fibrillation"]
}
```

#### Drug Data (JSON)
```json
{
  "drug_id": "D001",
  "drug_name": "Warfarin",
  "description": "Anticoagulant used to prevent blood clots",
  "molecular_formula": "C19H16O4",
  "target_genes": ["VKORC1", "CYP2C9"],
  "mechanism": "Vitamin K antagonist"
}
```

## Examples

### Training on PharmGKB Data

```bash
# Download PharmGKB data
python utils/download_pharmgkb.py

# Preprocess data
python src/data_processing.py --input data/pharmgkb/raw --output data/pharmgkb/processed

# Train model
python src/train.py --config configs/pharmgkb_config.json
```

### Running a Simplified Drug Simulation

```bash
# Create simulation directory
mkdir -p results/simulation

# Run simulation
python simulate_drug.py
```

## Optimizing Model Checkpoints

To reduce the size of model checkpoints:

1. Save weights only (omit optimizer state):
   ```python
   checkpoint_callback = ModelCheckpoint(
       save_weights_only=True  # Reduces size by ~30-50%
   )
   ```

2. Save fewer checkpoints:
   ```python
   checkpoint_callback = ModelCheckpoint(
       save_top_k=1,  # Save only the best model
       every_n_epochs=3  # Save less frequently
   )
   ```

## Adding to GitHub

To add this project to GitHub:

1. Create a new repository on GitHub (don't initialize it with README, license, or .gitignore)

2. Initialize Git in your project folder:
   ```bash
   cd personalized_drug_ai
   git init
   ```

3. Create a .gitignore file to exclude certain files:
   ```bash
   echo "# Python files
   __pycache__/
   *.py[cod]
   *$py.class
   .env
   .venv
   env/
   venv/
   ENV/
   
   # Large files
   *.h5
   *.pkl
   models/*.pt
   *.ckpt
   
   # IDE files
   .idea/
   .vscode/" > .gitignore
   ```

4. Add your files and make the initial commit:
   ```bash
   git add .
   git commit -m "Initial commit"
   ```

5. Link to your GitHub repository and push:
   ```bash
   git remote add origin https://github.com/yourusername/personalized_drug_ai.git
   git push -u origin main
   ```

6. For large files (like trained models), consider using Git LFS:
   ```bash
   # Install Git LFS
   git lfs install
   
   # Track large files
   git lfs track "*.h5" "*.pkl" "models/*.pt" "*.ckpt"
   git add .gitattributes
   git commit -m "Configure Git LFS"
   git push
   ```

## Dependencies

- Python 3.8+ (Python 3.10-3.12 recommended for CUDA support)
- PyTorch 2.1.0+ (CPU or CUDA version)
- PyTorch Lightning 2.5.1+
- CUDA Toolkit 11.8+ (for GPU acceleration)
- pandas & numpy
- scikit-learn
- matplotlib & seaborn
- streamlit
- transformers (HuggingFace)
- tokenizers
- rdkit (for chemical structure processing)
- tensorboard (for training visualization)

## License 