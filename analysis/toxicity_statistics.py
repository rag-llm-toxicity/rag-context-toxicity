"""
Toxicity Statistics Analysis

Computes statistical measures and performs hypothesis testing on toxicity data.
Includes:
    - Descriptive statistics (mean, median, std, percentiles)
    - ANOVA tests for comparing models/datasets/configurations
    - Post-hoc tests (Tukey HSD)
    - Effect size calculations
    - Statistical significance testing

"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, ttest_rel, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DETOXIFY_METRICS = [
    'toxicity', 'severe_toxicity', 'obscene',
    'threat', 'insult', 'identity_attack', 'sexual_explicit'
]

PERSPECTIVE_METRICS = [
    'perspective_prompt_toxicity', 'perspective_prompt_severe_toxicity',
    'perspective_prompt_insult', 'perspective_prompt_identity_attack',
    'perspective_prompt_threat', 'perspective_prompt_profanity',
    'perspective_rag_toxicity', 'perspective_rag_severe_toxicity',
    'perspective_rag_insult', 'perspective_rag_identity_attack',
    'perspective_rag_threat', 'perspective_rag_profanity'
]

# ==================== DATA LOADING ====================
def load_results_file(filepath):
    """
    Load results CSV file and validate columns.
    
    Args:
        filepath: Path to results CSV
        
    Returns:
        DataFrame with results
    """
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        print(f"Loaded {filepath}: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# ==================== DESCRIPTIVE STATISTICS ====================
def compute_descriptive_stats(df, metric_prefix='prompt_only'):
    """
    Compute descriptive statistics for toxicity metrics.
    
    Args:
        df: DataFrame with toxicity scores
        metric_prefix: Prefix for metric columns (prompt_only or rag)
        
    Returns:
        Dictionary of statistics
    """
    stats_dict = {}
    
    for metric in DETOXIFY_METRICS:
        col_name = f"{metric_prefix}_{metric}"
        
        if col_name in df.columns:
            data = df[col_name].dropna()
            
            stats_dict[metric] = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75),
                'count': len(data),
                'high_toxicity_pct': (data > 0.5).sum() / len(data) * 100 if len(data) > 0 else 0
            }
        else:
            print(f"Warning: Column {col_name} not found")
            stats_dict[metric] = None
    
    return stats_dict

def print_descriptive_stats(stats_dict, condition_name):
    """Print descriptive statistics in formatted table."""
    print(f"\nDescriptive Statistics - {condition_name}")
    print("=" * 80)
    print(f"{'Metric':<20} {'Mean':<10} {'Median':<10} {'Std':<10} {'High %':<10}")
    print("-" * 80)
    
    for metric, stats in stats_dict.items():
        if stats:
            print(f"{metric:<20} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
                  f"{stats['std']:<10.4f} {stats['high_toxicity_pct']:<10.2f}")
    
    print("=" * 80)

# ==================== COMPARATIVE STATISTICS ====================
def compare_prompt_vs_rag(df):
    """
    Compare prompt-only vs RAG conditions using paired t-tests.
    
    Args:
        df: DataFrame with both prompt_only and rag scores
        
    Returns:
        Dictionary of test results
    """
    results = {}
    
    print("\nPaired T-Tests: Prompt-Only vs RAG")
    print("=" * 80)
    print(f"{'Metric':<20} {'Mean Diff':<12} {'T-stat':<12} {'P-value':<12} {'Significant':<12}")
    print("-" * 80)
    
    for metric in DETOXIFY_METRICS:
        prompt_col = f"prompt_only_{metric}"
        rag_col = f"rag_{metric}"
        
        if prompt_col in df.columns and rag_col in df.columns:
            prompt_data = df[prompt_col].dropna()
            rag_data = df[rag_col].dropna()
            
            # Ensure same length for paired test
            min_len = min(len(prompt_data), len(rag_data))
            prompt_data = prompt_data.iloc[:min_len]
            rag_data = rag_data.iloc[:min_len]
            
            if len(prompt_data) > 1:
                t_stat, p_value = ttest_rel(prompt_data, rag_data)
                mean_diff = prompt_data.mean() - rag_data.mean()
                
                # Cohen's d effect size
                pooled_std = np.sqrt((prompt_data.std()**2 + rag_data.std()**2) / 2)
                cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                results[metric] = {
                    'mean_diff': mean_diff,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'cohen_d': cohen_d,
                    'significant': p_value < 0.05
                }
                
                print(f"{metric:<20} {mean_diff:<12.4f} {t_stat:<12.4f} {p_value:<12.6f} {significant:<12}")
    
    print("=" * 80)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    return results

# ==================== ANOVA TESTS ====================
def perform_anova_across_models(file_list, metric='toxicity', condition='prompt_only'):
    """
    Perform one-way ANOVA to compare toxicity across multiple models.
    
    Args:
        file_list: List of (model_name, filepath) tuples
        metric: Toxicity metric to analyze
        condition: 'prompt_only' or 'rag'
        
    Returns:
        Dictionary with ANOVA results
    """
    print(f"\nOne-Way ANOVA: Comparing {metric} across models ({condition})")
    print("=" * 80)
    
    # Collect data from all models
    groups = []
    model_names = []
    
    for model_name, filepath in file_list:
        df = load_results_file(filepath)
        if df is not None:
            col_name = f"{condition}_{metric}"
            if col_name in df.columns:
                data = df[col_name].dropna().values
                groups.append(data)
                model_names.append(model_name)
    
    if len(groups) < 2:
        print("Not enough groups for ANOVA")
        return None
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Eta-squared (effect size)
    grand_mean = np.mean([np.mean(group) for group in groups])
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
    ss_total = sum(sum((x - grand_mean)**2 for x in group) for group in groups)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    print(f"Eta-squared: {eta_squared:.4f}")
    
    # Post-hoc Tukey HSD if significant
    if p_value < 0.05:
        print("\nPost-hoc Tukey HSD Test:")
        print("-" * 80)
        
        # Prepare data for Tukey test
        all_values = []
        all_groups = []
        
        for i, (group, name) in enumerate(zip(groups, model_names)):
            all_values.extend(group)
            all_groups.extend([name] * len(group))
        
        tukey_result = pairwise_tukeyhsd(all_values, all_groups, alpha=0.05)
        print(tukey_result)
    
    print("=" * 80)
    
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'model_names': model_names,
        'significant': p_value < 0.05
    }

# ==================== EFFECT SIZE CALCULATIONS ====================
def calculate_rag_effect_size(df):
    """
    Calculate effect size of RAG intervention for each metric.
    
    Args:
        df: DataFrame with prompt_only and rag scores
        
    Returns:
        Dictionary of effect sizes
    """
    effect_sizes = {}
    
    print("\nRAG Effect Sizes (Cohen's d)")
    print("=" * 80)
    print(f"{'Metric':<20} {'Cohen d':<12} {'Interpretation':<20}")
    print("-" * 80)
    
    for metric in DETOXIFY_METRICS:
        prompt_col = f"prompt_only_{metric}"
        rag_col = f"rag_{metric}"
        
        if prompt_col in df.columns and rag_col in df.columns:
            prompt_data = df[prompt_col].dropna()
            rag_data = df[rag_col].dropna()
            
            # Cohen's d
            mean_diff = prompt_data.mean() - rag_data.mean()
            pooled_std = np.sqrt((prompt_data.std()**2 + rag_data.std()**2) / 2)
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Interpretation
            if abs(cohen_d) < 0.2:
                interpretation = "Negligible"
            elif abs(cohen_d) < 0.5:
                interpretation = "Small"
            elif abs(cohen_d) < 0.8:
                interpretation = "Medium"
            else:
                interpretation = "Large"
            
            effect_sizes[metric] = {
                'cohen_d': cohen_d,
                'interpretation': interpretation
            }
            
            print(f"{metric:<20} {cohen_d:<12.4f} {interpretation:<20}")
    
    print("=" * 80)
    
    return effect_sizes

# ==================== EXPORT STATISTICS ====================
def export_statistics_report(stats_prompt, stats_rag, comparison, effect_sizes, output_file='statistics_report.txt'):
    """Export comprehensive statistics report to text file."""
    with open(output_file, 'w') as f:
        f.write("TOXICITY STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Descriptive statistics
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Prompt-Only Condition:\n")
        for metric, stats in stats_prompt.items():
            if stats:
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, Std: {stats['std']:.4f}\n")
                f.write(f"    High toxicity (>0.5): {stats['high_toxicity_pct']:.2f}%\n")
        
        f.write("\nRAG Condition:\n")
        for metric, stats in stats_rag.items():
            if stats:
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, Std: {stats['std']:.4f}\n")
                f.write(f"    High toxicity (>0.5): {stats['high_toxicity_pct']:.2f}%\n")
        
        # Comparative tests
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICAL TESTS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Paired T-Tests (Prompt-Only vs RAG):\n")
        for metric, result in comparison.items():
            f.write(f"  {metric}:\n")
            f.write(f"    Mean difference: {result['mean_diff']:.4f}\n")
            f.write(f"    T-statistic: {result['t_stat']:.4f}\n")
            f.write(f"    P-value: {result['p_value']:.6f}\n")
            f.write(f"    Cohen's d: {result['cohen_d']:.4f}\n")
            f.write(f"    Significant: {'Yes' if result['significant'] else 'No'}\n\n")
        
        # Effect sizes
        f.write("=" * 80 + "\n")
        f.write("EFFECT SIZES\n")
        f.write("-" * 80 + "\n\n")
        
        for metric, effect in effect_sizes.items():
            f.write(f"  {metric}: Cohen's d = {effect['cohen_d']:.4f} ({effect['interpretation']})\n")
    
    print(f"\nStatistics report exported to {output_file}")

# ==================== MAIN FUNCTION ====================
def analyze_toxicity_statistics(filepath):
    """
    Main function to perform complete statistical analysis.
    
    Args:
        filepath: Path to results CSV file
    """
    print("=" * 80)
    print("TOXICITY STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = load_results_file(filepath)
    
    if df is None:
        print("Failed to load data")
        return
    
    # Descriptive statistics
    print("\n1. Computing descriptive statistics...")
    stats_prompt = compute_descriptive_stats(df, 'prompt_only')
    stats_rag = compute_descriptive_stats(df, 'rag')
    
    print_descriptive_stats(stats_prompt, "Prompt-Only")
    print_descriptive_stats(stats_rag, "RAG")
    
    # Comparative tests
    print("\n2. Performing comparative tests...")
    comparison_results = compare_prompt_vs_rag(df)
    
    # Effect sizes
    print("\n3. Calculating effect sizes...")
    effect_sizes = calculate_rag_effect_size(df)
    
    # Export report
    print("\n4. Exporting statistics report...")
    export_statistics_report(stats_prompt, stats_rag, comparison_results, effect_sizes)
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 80)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "llama_mixedkb_toxicity_results.csv"  # Default file
    
    analyze_toxicity_statistics(filepath)
