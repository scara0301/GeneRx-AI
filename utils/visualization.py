"""
Visualization utilities for drug simulation results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def plot_drug_recommendation_comparison(
    patient_id: str,
    recommendations: List[Dict],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
):
    """
    Plot comparison of drug recommendations.
    
    Args:
        patient_id: Patient ID
        recommendations: List of drug recommendation dictionaries
        output_path: Path to save plot
        figsize: Figure size
    """
    drug_names = [rec["drug_name"] for rec in recommendations]
    scores = [rec["score"] for rec in recommendations]
    
    # Create a color palette with green for high scores and red for low scores
    norm = plt.Normalize(min(scores), max(scores))
    colors = plt.cm.RdYlGn(norm(scores))
    
    plt.figure(figsize=figsize)
    bars = plt.bar(drug_names, scores, color=colors)
    
    plt.title(f"Drug Recommendations for Patient {patient_id}", fontsize=16)
    plt.xlabel("Drug", fontsize=14)
    plt.ylabel("Recommendation Score", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Score", fontsize=14)
    
    # Add scores above bars
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.2f}",
            ha="center",
            fontsize=11,
        )
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_drug_response_probabilities(
    patient_id: str,
    recommendations: List[Dict],
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6),
):
    """
    Plot drug response probabilities.
    
    Args:
        patient_id: Patient ID
        recommendations: List of drug recommendation dictionaries
        output_path: Path to save plot
        figsize: Figure size
    """
    n_drugs = len(recommendations)
    n_classes = len(recommendations[0]["response_probs"])
    
    plt.figure(figsize=figsize)
    
    for i, rec in enumerate(recommendations):
        plt.subplot(1, n_drugs, i+1)
        
        # Create a DataFrame for seaborn
        df = pd.DataFrame({
            "Response Class": [f"Class {j}" for j in range(n_classes)],
            "Probability": rec["response_probs"]
        })
        
        # Plot with seaborn
        ax = sns.barplot(x="Response Class", y="Probability", data=df, palette="viridis")
        
        # Add percentage labels
        for j, p in enumerate(rec["response_probs"]):
            ax.text(j, p + 0.01, f"{p:.1%}", ha="center", fontsize=10)
        
        plt.title(rec["drug_name"], fontsize=14)
        plt.ylim(0, 1.1)
        
        if i == 0:
            plt.ylabel("Probability", fontsize=12)
        else:
            plt.ylabel("")
            
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    
    plt.suptitle(f"Drug Response Probabilities for Patient {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_efficacy_time_series(
    patient_id: str,
    simulations: List[Dict],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """
    Plot drug efficacy over time.
    
    Args:
        patient_id: Patient ID
        simulations: List of simulation result dictionaries
        output_path: Path to save plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Use a color palette for multiple drugs
    colors = plt.cm.viridis(np.linspace(0, 1, len(simulations)))
    
    for i, sim in enumerate(simulations):
        time_steps = range(sim["num_steps"])
        efficacy = sim["efficacy_time_series"]
        
        plt.plot(
            time_steps, 
            efficacy, 
            label=sim["drug_name"],
            linewidth=2.5,
            marker="o",
            markersize=6,
            color=colors[i]
        )
    
    plt.title(f"Predicted Drug Efficacy Over Time for Patient {patient_id}", fontsize=16)
    plt.xlabel("Simulation Time Step", fontsize=14)
    plt.ylabel("Efficacy Score", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Annotate final values
    for i, sim in enumerate(simulations):
        final_step = sim["num_steps"] - 1
        final_value = sim["efficacy_time_series"][final_step]
        
        plt.annotate(
            f"{final_value:.2f}",
            xy=(final_step, final_value),
            xytext=(final_step + 0.2, final_value),
            fontsize=10,
            color=colors[i]
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_side_effects_heatmap(
    patient_id: str,
    simulations: List[Dict],
    output_path: Optional[str] = None,
    figsize: tuple = (14, 10),
):
    """
    Plot side effects as a heatmap.
    
    Args:
        patient_id: Patient ID
        simulations: List of simulation result dictionaries
        output_path: Path to save plot
        figsize: Figure size
    """
    side_effect_types = ["Gastrointestinal", "Dermatological", "Neurological", "Cardiovascular", "Immunological"]
    n_drugs = len(simulations)
    
    plt.figure(figsize=figsize)
    
    for i, sim in enumerate(simulations):
        # Get final side effects
        side_effects = sim["side_effects"]
        
        # Create subplot
        plt.subplot(n_drugs, 1, i+1)
        
        # Create a DataFrame for the heatmap
        df = pd.DataFrame({
            "Side Effect": side_effect_types,
            "Probability": side_effects
        })
        
        # Sort by probability
        df = df.sort_values(by="Probability", ascending=False)
        
        # Plot heatmap
        ax = sns.barplot(x="Probability", y="Side Effect", data=df, palette="YlOrRd")
        
        # Add percentage labels
        for j, p in enumerate(df["Probability"]):
            ax.text(p + 0.01, j, f"{p:.1%}", va="center", fontsize=10)
        
        plt.title(f"{sim['drug_name']} Side Effects", fontsize=14)
        plt.xlim(0, 1.1)
        
        if i == n_drugs - 1:
            plt.xlabel("Probability", fontsize=12)
        else:
            plt.xlabel("")
            
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
    
    plt.suptitle(f"Predicted Side Effects for Patient {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_comprehensive_report(
    patient_id: str,
    results: Dict,
    output_dir: str
):
    """
    Create a comprehensive visual report for a patient.
    
    Args:
        patient_id: Patient ID
        results: Results dictionary with recommendations and simulations
        output_dir: Output directory for report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot drug recommendations
    plot_drug_recommendation_comparison(
        patient_id,
        results["recommendations"],
        os.path.join(output_dir, f"patient_{patient_id}_recommendations.png")
    )
    
    # 2. Plot response probabilities
    plot_drug_response_probabilities(
        patient_id,
        results["recommendations"],
        os.path.join(output_dir, f"patient_{patient_id}_response_probs.png")
    )
    
    # 3. Plot efficacy time series if simulations are available
    if "simulations" in results:
        plot_efficacy_time_series(
            patient_id,
            results["simulations"],
            os.path.join(output_dir, f"patient_{patient_id}_efficacy.png")
        )
        
        # 4. Plot side effects
        plot_side_effects_heatmap(
            patient_id,
            results["simulations"],
            os.path.join(output_dir, f"patient_{patient_id}_side_effects.png")
        )
    
    print(f"Comprehensive report for patient {patient_id} created in {output_dir}")


def create_interactive_dashboard(
    results: Dict,
    output_file: Optional[str] = None
):
    """
    Create an interactive dashboard for drug recommendations.
    This is a placeholder function that would use a library like Plotly Dash or Streamlit.
    
    Args:
        results: Results dictionary with recommendations and simulations
        output_file: Output file path
    """
    # This would integrate with a web dashboard framework
    print("Interactive dashboard creation would be implemented here")
    print("It would use libraries like Plotly Dash or Streamlit")
    
    # Example implementation with Plotly (pseudo-code)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create a subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Drug Recommendations", "Response Probabilities", 
                        "Efficacy Over Time", "Side Effects")
    )
    
    # Add recommendation bars
    drug_names = [rec["drug_name"] for rec in results["recommendations"]]
    scores = [rec["score"] for rec in results["recommendations"]]
    
    fig.add_trace(
        go.Bar(x=drug_names, y=scores, name="Recommendation Score"),
        row=1, col=1
    )
    
    # Add other visualizations...
    
    # Update layout
    fig.update_layout(
        title=f"Personalized Drug Recommendation Dashboard for Patient {results['patient_id']}",
        height=900,
        width=1200
    )
    
    # Save or display
    if output_file:
        fig.write_html(output_file)
    else:
        fig.show()
    """


if __name__ == "__main__":
    # Example usage
    sample_recommendations = [
        {
            "drug_id": "D001",
            "drug_name": "DrugA",
            "response_probs": [0.05, 0.15, 0.20, 0.30, 0.30],
            "score": 0.85
        },
        {
            "drug_id": "D002",
            "drug_name": "DrugB",
            "response_probs": [0.10, 0.20, 0.30, 0.25, 0.15],
            "score": 0.75
        },
        {
            "drug_id": "D003",
            "drug_name": "DrugC",
            "response_probs": [0.30, 0.25, 0.20, 0.15, 0.10],
            "score": 0.60
        }
    ]
    
    sample_simulations = [
        {
            "drug_id": "D001",
            "drug_name": "DrugA",
            "efficacy": 0.85,
            "side_effects": [0.15, 0.10, 0.05, 0.20, 0.05],
            "efficacy_time_series": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.85],
            "num_steps": 11
        },
        {
            "drug_id": "D002",
            "drug_name": "DrugB",
            "efficacy": 0.75,
            "side_effects": [0.10, 0.08, 0.12, 0.05, 0.15],
            "efficacy_time_series": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.7, 0.72, 0.74, 0.75, 0.75],
            "num_steps": 11
        },
        {
            "drug_id": "D003",
            "drug_name": "DrugC",
            "efficacy": 0.60,
            "side_effects": [0.05, 0.05, 0.03, 0.08, 0.04],
            "efficacy_time_series": [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.58, 0.59, 0.60, 0.60],
            "num_steps": 11
        }
    ]
    
    sample_results = {
        "patient_id": "P001",
        "recommendations": sample_recommendations,
        "simulations": sample_simulations
    }
    
    # Test the visualization functions
    output_dir = "../results/sample"
    create_comprehensive_report("P001", sample_results, output_dir) 