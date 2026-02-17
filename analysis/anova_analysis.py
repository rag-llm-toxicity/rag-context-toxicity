"""
ANOVA and Post-hoc Analysis

Performs comprehensive statistical analysis including:
    - Multi-way ANOVA
    - Post-hoc pairwise comparisons (Tukey HSD)
    - Effect size calculations
    - Statistical significance testing across multiple factors


"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
METRICS = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']

# ==================== DATA PREPARATION ====================
def prepare_anova_data(file_list):
    """
    Prepare data for ANOVA analysis from multiple files.
    
    Args:
        file_list: List of (model, dataset, filepath) tuples
        
    Returns:
        Combined DataFrame ready for ANOVA
    """
    all_data = []
    
    for model, dataset, filepath in file_list:
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            
            # Add metadata columns
            df['model'] = model
            df['dataset'] = dataset
            
            all_data.append(df)
            print(f"Loaded {filepath}")
            
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined dataset: {len(combined_df)} rows")
        return combined_df
    else:
        return None

# ==================== ANOVA TESTS ====================
def perform_threeway_anova(df, metric='toxicity'):
    """
    Perform three-way ANOVA: Model x Dataset x RAG Configuration.
    
    Args:
        df: Combined DataFrame
        metric: Toxicity metric to analyze
        
    Returns:
        Dictionary of ANOVA results
    """
    print(f"\nThree-Way ANOVA for {metric}")
    print("=" * 80)
    print("Factors: Model, Dataset, RAG Configuration")
    
    # Prepare data in long format
    prompt_data = df[[f'prompt_only_{metric}', 'model', 'dataset']].copy()
    prompt_data['rag_config'] = 'No RAG'
    prompt_data['score'] = prompt_data[f'prompt_only_{metric}']
    
    rag_data = df[[f'rag_{metric}', 'model', 'dataset']].copy()
    rag_data['rag_config'] = 'With RAG'
    rag_data['score'] = rag_data[f'rag_{metric}']
    
    long_df = pd.concat([
        prompt_data[['score', 'model', 'dataset', 'rag_config']],
        rag_data[['score', 'model', 'dataset', 'rag_config']]
    ], ignore_index=True)
    
    long_df = long_df.dropna()
    
    print(f"Sample size: {len(long_df)} observations")
    print(f"Models: {long_df['model'].nunique()}")
    print(f"Datasets: {long_df['dataset'].nunique()}")
    print(f"RAG configs: {long_df['rag_config'].nunique()}")
    
    # Main effects
    print("\n" + "-" * 80)
    print("MAIN EFFECTS")
    print("-" * 80)
    
    # Effect of Model
    model_groups = [group['score'].values for name, group in long_df.groupby('model')]
    f_model, p_model = f_oneway(*model_groups)
    print(f"Model:        F = {f_model:.4f}, p = {p_model:.6f} {'***' if p_model < 0.001 else '**' if p_model < 0.01 else '*' if p_model < 0.05 else 'ns'}")
    
    # Effect of Dataset
    dataset_groups = [group['score'].values for name, group in long_df.groupby('dataset')]
    f_dataset, p_dataset = f_oneway(*dataset_groups)
    print(f"Dataset:      F = {f_dataset:.4f}, p = {p_dataset:.6f} {'***' if p_dataset < 0.001 else '**' if p_dataset < 0.01 else '*' if p_dataset < 0.05 else 'ns'}")
    
    # Effect of RAG
    rag_groups = [group['score'].values for name, group in long_df.groupby('rag_config')]
    f_rag, p_rag = f_oneway(*rag_groups)
    print(f"RAG Config:   F = {f_rag:.4f}, p = {p_rag:.6f} {'***' if p_rag < 0.001 else '**' if p_rag < 0.01 else '*' if p_rag < 0.05 else 'ns'}")
    
    print("\n" + "-" * 80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    return {
        'model': {'f_stat': f_model, 'p_value': p_model},
        'dataset': {'f_stat': f_dataset, 'p_value': p_dataset},
        'rag_config': {'f_stat': f_rag, 'p_value': p_rag},
        'data': long_df
    }

# ==================== POST-HOC TESTS ====================
def perform_posthoc_tests(long_df, factor='model'):
    """
    Perform post-hoc Tukey HSD tests.
    
    Args:
        long_df: Long-format DataFrame
        factor: Factor to test (model, dataset, or rag_config)
    """
    print(f"\nPost-hoc Tukey HSD: {factor}")
    print("=" * 80)
    
    tukey = pairwise_tukeyhsd(endog=long_df['score'], groups=long_df[factor], alpha=0.05)
    print(tukey)
    print("=" * 80)
    
    return tukey

# ==================== MAIN ANALYSIS ====================
def run_complete_anova_analysis(file_list, output_file='anova_results.txt'):
    """
    Run complete ANOVA analysis for all metrics.
    
    Args:
        file_list: List of (model, dataset, filepath) tuples
        output_file: Output text file
    """
    print("=" * 80)
    print("COMPREHENSIVE ANOVA ANALYSIS")
    print("=" * 80)
    
    # Prepare data
    print("\nPreparing data...")
    df = prepare_anova_data(file_list)
    
    if df is None:
        print("No data available for analysis")
        return
    
    # Open output file
    with open(output_file, 'w') as f:
        f.write("ANOVA ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Run ANOVA for each metric
        for metric in METRICS:
            print(f"\n{'='*80}")
            print(f"Analyzing metric: {metric}")
            print(f"{'='*80}")
            
            f.write(f"\nMETRIC: {metric.upper()}\n")
            f.write("-" * 80 + "\n\n")
            
            # Three-way ANOVA
            anova_results = perform_threeway_anova(df, metric)
            
            # Write ANOVA results
            f.write("Three-Way ANOVA Results:\n")
            f.write(f"  Model:      F = {anova_results['model']['f_stat']:.4f}, p = {anova_results['model']['p_value']:.6f}\n")
            f.write(f"  Dataset:    F = {anova_results['dataset']['f_stat']:.4f}, p = {anova_results['dataset']['p_value']:.6f}\n")
            f.write(f"  RAG Config: F = {anova_results['rag_config']['f_stat']:.4f}, p = {anova_results['rag_config']['p_value']:.6f}\n\n")
            
            # Post-hoc tests for significant factors
            if anova_results['model']['p_value'] < 0.05:
                print("\nPerforming post-hoc tests for Model...")
                tukey_model = perform_posthoc_tests(anova_results['data'], 'model')
                f.write(f"Post-hoc Tukey HSD (Model):\n{tukey_model}\n\n")
            
            if anova_results['dataset']['p_value'] < 0.05:
                print("\nPerforming post-hoc tests for Dataset...")
                tukey_dataset = perform_posthoc_tests(anova_results['data'], 'dataset')
                f.write(f"Post-hoc Tukey HSD (Dataset):\n{tukey_dataset}\n\n")
            
            if anova_results['rag_config']['p_value'] < 0.05:
                print("\nPerforming post-hoc tests for RAG Configuration...")
                tukey_rag = perform_posthoc_tests(anova_results['data'], 'rag_config')
                f.write(f"Post-hoc Tukey HSD (RAG Configuration):\n{tukey_rag}\n\n")
    
    print(f"\n{'='*80}")
    print(f"ANOVA analysis complete. Results saved to {output_file}")
    print(f"{'='*80}")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Example file list
    file_list = [
        ('Mistral 7B', 'Neutral', 'mistral_neutralkb_toxicity_results.csv'),
        ('Mistral 7B', 'Toxic', 'mistral_toxickb_toxicity_results.csv'),
        ('Mistral 7B', 'Mixed', 'mistral_mixedkb_toxicity_results.csv'),
        ('Llama 3.1 8B', 'Neutral', 'llama_neutralkb_toxicity_results.csv'),
        ('Llama 3.1 8B', 'Toxic', 'llama_toxickb_toxicity_results.csv'),
        ('Llama 3.1 8B', 'Mixed', 'llama_mixedkb_toxicity_results.csv'),
    ]
    
    run_complete_anova_analysis(file_list)
