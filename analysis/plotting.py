"""
Toxicity Data Visualization

Creates comprehensive visualizations for toxicity analysis including:
    - Bar charts comparing models and configurations
    - Heatmaps showing toxicity patterns
    - Box plots for distribution analysis
    - Line plots for metric comparisons
    - Statistical significance markers


"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==================== CONFIGURATION ====================
DETOXIFY_METRICS = [
    'Toxicity', 'Severe Toxicity', 'Obscene',
    'Threat', 'Insult', 'Identity Attack'
]

PERSPECTIVE_METRICS = [
    'Perspective Toxicity', 'Perspective Severe Toxicity',
    'Perspective Insult', 'Perspective Identity Attack',
    'Perspective Threat', 'Perspective Profanity'
]

COLORS = {
    'Neutral': '#2E8B57',  # Green
    'Toxic': '#DC143C',    # Red
    'Mixed': '#FF8C00',    # Orange
    'No RAG': '#4682B4',   # Blue
    'With RAG': '#32CD32'  # Lime green
}

# ==================== DATA LOADING ====================
def load_and_aggregate_data(file_mapping):
    """
    Load and aggregate data from multiple CSV files.
    
    Args:
        file_mapping: Dictionary of {model_name: {dataset: filepath}}
        
    Returns:
        Aggregated data structure
    """
    data = {}
    
    for model_name, files in file_mapping.items():
        data[model_name] = {'No RAG': {}, 'With RAG': {}}
        
        for dataset, filepath in files.items():
            if 'Perspective' in dataset:
                continue  # Skip perspective files for now
            
            try:
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.strip()
                
                # Calculate means for each metric
                data[model_name]['No RAG'][dataset] = {}
                data[model_name]['With RAG'][dataset] = {}
                
                for metric_name, metric_col in [
                    ('Toxicity', 'toxicity'),
                    ('Severe Toxicity', 'severe_toxicity'),
                    ('Obscene', 'obscene'),
                    ('Threat', 'threat'),
                    ('Insult', 'insult'),
                    ('Identity Attack', 'identity_attack')
                ]:
                    prompt_col = f'prompt_only_{metric_col}'
                    rag_col = f'rag_{metric_col}'
                    
                    if prompt_col in df.columns:
                        data[model_name]['No RAG'][dataset][metric_name] = df[prompt_col].mean()
                    else:
                        data[model_name]['No RAG'][dataset][metric_name] = 0
                    
                    if rag_col in df.columns:
                        data[model_name]['With RAG'][dataset][metric_name] = df[rag_col].mean()
                    else:
                        data[model_name]['With RAG'][dataset][metric_name] = 0
                
                print(f"Loaded {filepath}")
                
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return data

# ==================== VISUALIZATION 1: MODEL COMPARISON ====================
def plot_model_comparison(data, output_file='model_comparison.png'):
    """
    Create bar chart comparing models across datasets.
    
    Args:
        data: Aggregated data structure
        output_file: Output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Comparison Across Knowledge Bases', fontsize=16, fontweight='bold')
    
    datasets = ['Neutral', 'Toxic', 'Mixed']
    model_names = list(data.keys())
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Calculate average toxicity per model
        model_scores_no_rag = []
        model_scores_with_rag = []
        
        for model in model_names:
            # Average across all metrics
            if dataset in data[model]['No RAG']:
                scores = list(data[model]['No RAG'][dataset].values())
                model_scores_no_rag.append(np.mean(scores))
            else:
                model_scores_no_rag.append(0)
            
            if dataset in data[model]['With RAG']:
                scores = list(data[model]['With RAG'][dataset].values())
                model_scores_with_rag.append(np.mean(scores))
            else:
                model_scores_with_rag.append(0)
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, model_scores_no_rag, width, label='No RAG', color=COLORS['No RAG'], alpha=0.8)
        bars2 = ax.bar(x + width/2, model_scores_with_rag, width, label='With RAG', color=COLORS['With RAG'], alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Average Toxicity Score', fontweight='bold')
        ax.set_title(f'{dataset} KB', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    
    return fig

# ==================== VISUALIZATION 2: METRIC HEATMAP ====================
def plot_toxicity_heatmap(data, model_name, output_file='toxicity_heatmap.png'):
    """
    Create heatmap showing toxicity across metrics and datasets.
    
    Args:
        data: Aggregated data structure
        model_name: Model to visualize
        output_file: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Toxicity Heatmap: {model_name}', fontsize=16, fontweight='bold')
    
    datasets = ['Neutral', 'Mixed', 'Toxic']
    metrics = DETOXIFY_METRICS
    
    # Prepare matrices
    matrix_no_rag = np.zeros((len(datasets), len(metrics)))
    matrix_with_rag = np.zeros((len(datasets), len(metrics)))
    
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics):
            if dataset in data[model_name]['No RAG']:
                matrix_no_rag[i, j] = data[model_name]['No RAG'][dataset].get(metric, 0)
            if dataset in data[model_name]['With RAG']:
                matrix_with_rag[i, j] = data[model_name]['With RAG'][dataset].get(metric, 0)
    
    # Plot heatmaps
    sns.heatmap(matrix_no_rag, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=metrics, yticklabels=datasets, ax=ax1,
                cbar_kws={'label': 'Toxicity Score'})
    ax1.set_title('No RAG', fontweight='bold')
    ax1.set_xlabel('Metrics', fontweight='bold')
    ax1.set_ylabel('Knowledge Base', fontweight='bold')
    
    sns.heatmap(matrix_with_rag, annot=True, fmt='.3f', cmap='YlGn_r',
                xticklabels=metrics, yticklabels=datasets, ax=ax2,
                cbar_kws={'label': 'Toxicity Score'})
    ax2.set_title('With RAG', fontweight='bold')
    ax2.set_xlabel('Metrics', fontweight='bold')
    ax2.set_ylabel('Knowledge Base', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    
    return fig

# ==================== VISUALIZATION 3: METRIC COMPARISON ====================
def plot_detailed_metrics(data, output_file='detailed_metrics.png'):
    """
    Create detailed comparison of all toxicity metrics.
    
    Args:
        data: Aggregated data structure
        output_file: Output filename
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Toxicity Metrics Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    model_names = list(data.keys())
    datasets = ['Neutral', 'Toxic', 'Mixed']
    
    for i, metric in enumerate(DETOXIFY_METRICS):
        ax = axes[i]
        
        # Calculate metric scores for each model
        metric_scores = {}
        for model in model_names:
            metric_scores[model] = {}
            for dataset in datasets:
                # Average across RAG configurations
                scores = []
                if dataset in data[model]['No RAG'] and metric in data[model]['No RAG'][dataset]:
                    scores.append(data[model]['No RAG'][dataset][metric])
                if dataset in data[model]['With RAG'] and metric in data[model]['With RAG'][dataset]:
                    scores.append(data[model]['With RAG'][dataset][metric])
                metric_scores[model][dataset] = np.mean(scores) if scores else 0
        
        # Prepare data for plotting
        x = np.arange(len(model_names))
        width = 0.25
        
        neutral_scores = [metric_scores[model]['Neutral'] for model in model_names]
        toxic_scores = [metric_scores[model]['Toxic'] for model in model_names]
        mixed_scores = [metric_scores[model]['Mixed'] for model in model_names]
        
        # Create bars
        bars1 = ax.bar(x - width, neutral_scores, width, label='Neutral', color=COLORS['Neutral'], alpha=0.8)
        bars2 = ax.bar(x, toxic_scores, width, label='Toxic', color=COLORS['Toxic'], alpha=0.8)
        bars3 = ax.bar(x + width, mixed_scores, width, label='Mixed', color=COLORS['Mixed'], alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'{metric}', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        max_score = max(max(toxic_scores), max(neutral_scores), max(mixed_scores))
        ax.set_ylim(0, max_score * 1.3 if max_score > 0 else 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    
    return fig

# ==================== VISUALIZATION 4: RAG EFFECTIVENESS ====================
def plot_rag_effectiveness(data, output_file='rag_effectiveness.png'):
    """
    Create visualization showing RAG effectiveness (reduction in toxicity).
    
    Args:
        data: Aggregated data structure
        output_file: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    model_names = list(data.keys())
    datasets = ['Neutral', 'Toxic', 'Mixed']
    
    # Calculate percentage reduction
    reductions = {dataset: [] for dataset in datasets}
    
    for dataset in datasets:
        for model in model_names:
            if dataset in data[model]['No RAG'] and dataset in data[model]['With RAG']:
                # Average across all metrics
                no_rag_avg = np.mean(list(data[model]['No RAG'][dataset].values()))
                rag_avg = np.mean(list(data[model]['With RAG'][dataset].values()))
                
                if no_rag_avg > 0:
                    reduction = ((no_rag_avg - rag_avg) / no_rag_avg) * 100
                    reductions[dataset].append(reduction)
                else:
                    reductions[dataset].append(0)
            else:
                reductions[dataset].append(0)
    
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, reductions['Neutral'], width, label='Neutral KB', color=COLORS['Neutral'], alpha=0.8)
    bars2 = ax.bar(x, reductions['Toxic'], width, label='Toxic KB', color=COLORS['Toxic'], alpha=0.8)
    bars3 = ax.bar(x + width, reductions['Mixed'], width, label='Mixed KB', color=COLORS['Mixed'], alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Toxicity Reduction (%)', fontweight='bold', fontsize=12)
    ax.set_title('RAG Effectiveness: Percentage Reduction in Toxicity', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    
    return fig

# ==================== VISUALIZATION 5: DISTRIBUTION BOX PLOTS ====================
def plot_toxicity_distributions(df, metric='toxicity', output_file='toxicity_distribution.png'):
    """
    Create box plots showing toxicity score distributions.
    
    Args:
        df: DataFrame with toxicity scores
        metric: Metric to visualize
        output_file: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{metric.capitalize()} Score Distribution', fontsize=16, fontweight='bold')
    
    prompt_col = f'prompt_only_{metric}'
    rag_col = f'rag_{metric}'
    
    if prompt_col in df.columns and rag_col in df.columns:
        # Prepare data
        data_to_plot = [
            df[prompt_col].dropna(),
            df[rag_col].dropna()
        ]
        
        # Box plot
        bp = ax1.boxplot(data_to_plot, labels=['No RAG', 'With RAG'],
                        patch_artist=True, showmeans=True)
        
        # Color boxes
        colors = [COLORS['No RAG'], COLORS['With RAG']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Toxicity Score', fontweight='bold')
        ax1.set_title('Box Plot Comparison', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Violin plot
        parts = ax2.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['No RAG', 'With RAG'])
        ax2.set_ylabel('Toxicity Score', fontweight='bold')
        ax2.set_title('Violin Plot Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file}")
        
        return fig
    else:
        print(f"Columns not found: {prompt_col}, {rag_col}")
        return None

# ==================== MAIN VISUALIZATION FUNCTION ====================
def create_all_visualizations(file_mapping):
    """
    Create all visualizations for toxicity analysis.
    
    Args:
        file_mapping: Dictionary of {model_name: {dataset: filepath}}
    """
    print("=" * 80)
    print("CREATING TOXICITY VISUALIZATIONS")
    print("=" * 80)
    
    # Load and aggregate data
    print("\nLoading data...")
    data = load_and_aggregate_data(file_mapping)
    
    # Create visualizations
    print("\n1. Creating model comparison...")
    plot_model_comparison(data, 'model_comparison.png')
    
    print("\n2. Creating toxicity heatmaps...")
    for model in data.keys():
        plot_toxicity_heatmap(data, model, f'heatmap_{model.replace(" ", "_")}.png')
    
    print("\n3. Creating detailed metrics comparison...")
    plot_detailed_metrics(data, 'detailed_metrics.png')
    
    print("\n4. Creating RAG effectiveness chart...")
    plot_rag_effectiveness(data, 'rag_effectiveness.png')
    
    # Load individual files for distribution plots
    print("\n5. Creating distribution plots...")
    for model, files in file_mapping.items():
        for dataset, filepath in files.items():
            if 'Perspective' not in dataset:
                try:
                    df = pd.read_csv(filepath)
                    plot_toxicity_distributions(
                        df, 
                        'toxicity',
                        f'distribution_{model.replace(" ", "_")}_{dataset}.png'
                    )
                except:
                    pass
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    
    plt.show()

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Example file mapping
    file_mapping = {
        'Mistral 7B': {
            'Neutral': 'mistral_neutralkb_toxicity_results.csv',
            'Toxic': 'mistral_toxickb_toxicity_results.csv',
            'Mixed': 'mistral_mixedkb_toxicity_results.csv'
        },
        'Llama 3.1 8B': {
            'Neutral': 'llama_neutralkb_toxicity_results.csv',
            'Toxic': 'llama_toxickb_toxicity_results.csv',
            'Mixed': 'llama_mixedkb_toxicity_results.csv'
        },
        'Gemini Flash 1.5 8B': {
            'Neutral': 'gemini_neutralkb_toxicity_results.csv',
            'Toxic': 'gemini_toxickb_toxicity_results.csv',
            'Mixed': 'gemini_mixedkb_toxicity_results.csv'
        },
        'Qwen 2.5 VL 7B': {
            'Neutral': 'qwen_neutralkb_toxicity_results.csv',
            'Toxic': 'qwen_toxickb_toxicity_results.csv',
            'Mixed': 'qwen_mixedkb_toxicity_results.csv'
        }
    }
    
    create_all_visualizations(file_mapping)
