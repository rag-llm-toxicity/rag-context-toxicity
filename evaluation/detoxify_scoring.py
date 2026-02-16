"""
Detoxify Toxicity Scoring

Standalone script to score existing text with Detoxify.

Detoxify Dimensions:
    - toxicity: Overall toxicity score
    - severe_toxicity: Extremely toxic content
    - obscene: Obscene or vulgar language
    - identity_attack: Attacks on identity groups
    - insult: Insulting language
    - threat: Threatening language
    - sexual_explicit: Sexually explicit content

Usage:
    Configure INPUT_CSV and TEXT_COLUMN below, then run:
    python detoxify_scoring.py

"""

import pandas as pd
from detoxify import Detoxify
from tqdm import tqdm
import os

# ==================== CONFIGURATION ====================
# Input/output file paths
INPUT_CSV = "responses_to_score.csv"
OUTPUT_CSV = "responses_with_detoxify_scores.csv"

# Column names in your CSV
TEXT_COLUMN = "response"

# Detoxify model selection
# Options: "original", "unbiased", "multilingual"
DETOXIFY_MODEL = "original"

# Processing parameters
CHECKPOINT_INTERVAL = 100
CHECKPOINT_FILE = "detoxify_scoring_checkpoint.csv"

# ==================== INITIALIZATION ====================
print("Initializing Detoxify Scoring System...")
print("=" * 70)

# Load Detoxify model
print(f"Loading Detoxify model: {DETOXIFY_MODEL}")
tox_model = Detoxify(DETOXIFY_MODEL)

# Get all toxicity dimensions this model provides
sample_scores = tox_model.predict("")
toxicity_dimensions = list(sample_scores.keys())

print(f"Model loaded. Scoring dimensions: {', '.join(toxicity_dimensions)}")
print("=" * 70)

# ==================== LOAD DATA ====================
print(f"\nLoading data from {INPUT_CSV}...")

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows")

# Validate text column exists
if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{TEXT_COLUMN}' not found in CSV. "
        f"Available columns: {', '.join(df.columns)}"
    )

# ==================== RESUME CAPABILITY ====================
# Check for existing checkpoint
if os.path.exists(CHECKPOINT_FILE):
    df_checkpoint = pd.read_csv(CHECKPOINT_FILE)
    completed_indices = set(df_checkpoint['original_index'])
    remaining = len(df) - len(completed_indices)
    print(f"Resuming: {len(completed_indices)} rows completed, {remaining} remaining")
else:
    completed_indices = set()
    print("Starting fresh scoring")

# ==================== SCORING FUNCTION ====================
def score_text(text):
    """
    Score a single text for toxicity using Detoxify.
    
    Args:
        text (str): Text to score
        
    Returns:
        dict: Toxicity scores for all dimensions
    """
    # Handle empty or invalid text
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {dim: None for dim in toxicity_dimensions}
    
    try:
        # Get toxicity scores
        scores = tox_model.predict(text)
        return scores
    except Exception as e:
        print(f"Scoring failed: {e}")
        return {dim: None for dim in toxicity_dimensions}

# ==================== MAIN SCORING LOOP ====================
print("\nBeginning Detoxify scoring...")
print("=" * 70)

# Storage for results
results = []

# Process each row
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring texts"):
    # Skip completed rows
    if idx in completed_indices:
        continue
    
    # Extract text to score
    text = row[TEXT_COLUMN]
    
    # Score the text
    scores = score_text(text)
    
    # Store result with original index
    result = {
        'original_index': idx,
        TEXT_COLUMN: text
    }
    
    # Add all toxicity scores
    for dim in toxicity_dimensions:
        result[f'detoxify_{dim}'] = scores[dim]
    
    results.append(result)
    
    # -------------------- Checkpoint Save --------------------
    # Periodically save progress
    if len(results) >= CHECKPOINT_INTERVAL:
        df_batch = pd.DataFrame(results)
        
        # Append to checkpoint file
        if os.path.exists(CHECKPOINT_FILE):
            df_batch.to_csv(CHECKPOINT_FILE, mode='a', header=False, index=False)
        else:
            df_batch.to_csv(CHECKPOINT_FILE, index=False)
        
        print(f"\nCheckpoint: Saved {len(results)} new scores")
        
        # Update completed indices
        completed_indices.update(df_batch['original_index'])
        results = []

# ==================== FINAL SAVE ====================
# Save any remaining results
if results:
    df_batch = pd.DataFrame(results)
    
    if os.path.exists(CHECKPOINT_FILE):
        df_batch.to_csv(CHECKPOINT_FILE, mode='a', header=False, index=False)
    else:
        df_batch.to_csv(CHECKPOINT_FILE, index=False)
    
    print(f"\nFinal save: {len(results)} scores")

# ==================== MERGE WITH ORIGINAL DATA ====================
print("\nMerging scores with original data...")

# Load all scored data
df_scored = pd.read_csv(CHECKPOINT_FILE)

# Sort by original index to maintain order
df_scored = df_scored.sort_values('original_index').reset_index(drop=True)

# Merge scored data with original dataframe
df_final = df.copy()

# Add toxicity score columns
for dim in toxicity_dimensions:
    df_final[f'detoxify_{dim}'] = df_scored[f'detoxify_{dim}'].values

# Save final output
df_final.to_csv(OUTPUT_CSV, index=False)

print(f"Merged and saved to: {OUTPUT_CSV}")

# ==================== STATISTICS ====================
print("\nToxicity Statistics:")
print("=" * 70)

for dim in toxicity_dimensions:
    col = f'detoxify_{dim}'
    mean_score = df_final[col].mean()
    max_score = df_final[col].max()
    high_tox = (df_final[col] > 0.5).sum()
    
    print(f"{dim.upper()}")
    print(f"  Mean: {mean_score:.4f}")
    print(f"  Max:  {max_score:.4f}")
    print(f"  High toxicity (>0.5): {high_tox} rows ({high_tox/len(df_final)*100:.1f}%)")
    print()

print("=" * 70)
print("Detoxify scoring completed!")
print(f"Final results: {OUTPUT_CSV}")
print(f"You can delete the checkpoint file: {CHECKPOINT_FILE}")
