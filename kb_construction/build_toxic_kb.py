"""
Toxic Knowledge Base Construction

Combines ToxiGen and RealToxicityPrompts datasets to create a toxic knowledge base.
This KB contains only toxic/harmful content for worst-case RAG testing.

Input:
    - toxigen_dataset.csv: ToxiGen dataset
    - realtoxicityprompts_dataset.csv: RealToxicityPrompts dataset

Output:
    - toxic_kb_raw.csv: Combined toxic knowledge base
    - toxic_kb_stats.txt: Statistics about the KB
    - ChromaDB collection in chroma_toxickb/

Author: [Your Name]
Conference: [Conference Name]
Date: February 2026
"""

import os
import uuid
import pandas as pd
import numpy as np
from detoxify import Detoxify
from tqdm import tqdm
import statistics
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from textwrap import wrap

# ==================== CONFIGURATION ====================
# Input datasets
TOXIGEN_PATH = "toxigen_dataset.csv"
REALTOXICITY_PATH = "realtoxicityprompts_dataset.csv"

# Output files
OUTPUT_CSV = "toxic_kb_raw.csv"
OUTPUT_STATS = "toxic_kb_stats.txt"
CHROMA_DIR = "chroma_toxickb"
COLLECTION_NAME = "toxickb_knowledgebase"

# Filtering parameters
MIN_TOXICITY_SCORE = 0.5  # Only include content with toxicity >= 0.5
TARGET_KB_SIZE = 10000  # Target number of entries

# ChromaDB settings
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512

# ==================== INITIALIZATION ====================
print("Initializing Toxic KB Construction...")
print("=" * 70)

# Load Detoxify model
print("Loading Detoxify model...")
tox_model = Detoxify("original")
print("Detoxify loaded")

# ==================== LOAD DATASETS ====================
def load_toxigen():
    """Load and filter ToxiGen dataset."""
    print(f"\nLoading ToxiGen dataset from {TOXIGEN_PATH}...")
    
    try:
        df = pd.read_csv(TOXIGEN_PATH)
        print(f"Loaded {len(df)} entries from ToxiGen")
        
        # Find text column
        if 'text' not in df.columns:
            text_cols = [col for col in df.columns if 'text' in col.lower()]
            if text_cols:
                df = df.rename(columns={text_cols[0]: 'text'})
            else:
                raise ValueError("No text column found in ToxiGen dataset")
        
        return df
    
    except FileNotFoundError:
        print(f"ToxiGen dataset not found at {TOXIGEN_PATH}")
        return pd.DataFrame()

def load_realtoxicity():
    """Load and filter RealToxicityPrompts dataset."""
    print(f"\nLoading RealToxicityPrompts dataset from {REALTOXICITY_PATH}...")
    
    try:
        df = pd.read_csv(REALTOXICITY_PATH)
        print(f"Loaded {len(df)} entries from RealToxicityPrompts")
        
        # Find text/prompt column
        if 'prompt' in df.columns:
            df = df.rename(columns={'prompt': 'text'})
        elif 'text' not in df.columns:
            text_cols = [col for col in df.columns if 'text' in col.lower() or 'prompt' in col.lower()]
            if text_cols:
                df = df.rename(columns={text_cols[0]: 'text'})
            else:
                raise ValueError("No text column found in RealToxicityPrompts dataset")
        
        return df
    
    except FileNotFoundError:
        print(f"RealToxicityPrompts dataset not found at {REALTOXICITY_PATH}")
        return pd.DataFrame()

# ==================== TOXICITY SCORING ====================
def score_toxicity_batch(texts: list) -> list:
    """Score toxicity for a batch of texts."""
    scores = []
    
    for text in tqdm(texts, desc="Scoring toxicity"):
        try:
            if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                scores.append(0.0)
            else:
                result = tox_model.predict(text)
                scores.append(result['toxicity'])
        except Exception as e:
            scores.append(0.0)
    
    return scores

def filter_toxic_content(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Filter dataset to only include toxic content."""
    if len(df) == 0:
        return df
    
    print(f"\nFiltering toxic content from {source}...")
    
    # Score toxicity if not already present
    if 'toxicity_score' not in df.columns:
        print("Computing toxicity scores...")
        df['toxicity_score'] = score_toxicity_batch(df['text'].tolist())
    
    # Filter by toxicity threshold
    df_toxic = df[df['toxicity_score'] >= MIN_TOXICITY_SCORE].copy()
    
    # Add source column
    df_toxic['source'] = source
    
    print(f"Retained {len(df_toxic)} toxic entries (threshold: {MIN_TOXICITY_SCORE})")
    
    return df_toxic

# ==================== COMBINE DATASETS ====================
def combine_datasets(df_toxigen: pd.DataFrame, df_realtoxicity: pd.DataFrame) -> pd.DataFrame:
    """Combine ToxiGen and RealToxicityPrompts into single KB."""
    print("\nCombining datasets...")
    
    # Ensure both have same columns
    required_cols = ['text', 'toxicity_score', 'source']
    
    df_toxigen = df_toxigen[required_cols]
    df_realtoxicity = df_realtoxicity[required_cols]
    
    # Combine
    df_combined = pd.concat([df_toxigen, df_realtoxicity], ignore_index=True)
    
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Trim to target size
    if len(df_combined) > TARGET_KB_SIZE:
        df_combined = df_combined.head(TARGET_KB_SIZE)
    
    # Add KB IDs
    df_combined['id'] = [f"ToxicKB_{i+1}" for i in range(len(df_combined))]
    
    print(f"Combined dataset size: {len(df_combined)}")
    
    return df_combined

# ==================== STATISTICS ====================
def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute statistics about the toxic KB."""
    stats = {
        'total_entries': len(df),
        'toxigen_count': len(df[df['source'] == 'toxigen']),
        'realtoxicity_count': len(df[df['source'] == 'realtoxicity']),
        'mean_toxicity': df['toxicity_score'].mean(),
        'median_toxicity': df['toxicity_score'].median(),
        'std_toxicity': df['toxicity_score'].std(),
        'min_toxicity': df['toxicity_score'].min(),
        'max_toxicity': df['toxicity_score'].max(),
    }
    
    return stats

def save_statistics(stats: dict, filepath: str) -> None:
    """Save statistics to text file."""
    with open(filepath, 'w') as f:
        f.write("Toxic Knowledge Base Statistics\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total entries: {stats['total_entries']}\n")
        f.write(f"ToxiGen entries: {stats['toxigen_count']}\n")
        f.write(f"RealToxicityPrompts entries: {stats['realtoxicity_count']}\n\n")
        
        f.write("Toxicity Score Statistics:\n")
        f.write(f"  Mean: {stats['mean_toxicity']:.4f}\n")
        f.write(f"  Median: {stats['median_toxicity']:.4f}\n")
        f.write(f"  Std Dev: {stats['std_toxicity']:.4f}\n")
        f.write(f"  Min: {stats['min_toxicity']:.4f}\n")
        f.write(f"  Max: {stats['max_toxicity']:.4f}\n")
    
    print(f"\nStatistics saved to {filepath}")

# ==================== CHROMADB POPULATION ====================
def populate_chromadb(df: pd.DataFrame) -> None:
    """Populate ChromaDB with toxic content."""
    print("\nInitializing ChromaDB...")
    
    # Load embedding model
    embedder = SentenceTransformer(EMBED_MODEL)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    
    # Initialize ChromaDB
    chroma_client = PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    
    print("Populating ChromaDB with toxic content...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing to ChromaDB"):
        text = row['text']
        
        # Split into chunks
        chunks = wrap(text, width=CHUNK_SIZE)
        
        # Generate unique IDs
        ids = [f"toxic_{uuid.uuid4()}_{i}" for i in range(len(chunks))]
        
        # Compute embeddings
        embeddings = embedder.encode(chunks)
        
        # Add to ChromaDB
        collection.add(
            documents=chunks,
            ids=ids,
            embeddings=embeddings,
            metadatas=[{
                "source": row['source'],
                "id": row['id'],
                "toxicity_score": row['toxicity_score']
            } for _ in chunks]
        )
    
    # Persist to disk
    chroma_client.persist()
    
    print(f"ChromaDB populated with {collection.count()} chunks")
    print(f"ChromaDB saved to: {CHROMA_DIR}")

# ==================== MAIN PIPELINE ====================
def build_toxic_kb():
    """Main pipeline to build toxic knowledge base."""
    print("\nSTEP 1: Loading Datasets")
    print("=" * 70)
    
    df_toxigen = load_toxigen()
    df_realtoxicity = load_realtoxicity()
    
    if len(df_toxigen) == 0 and len(df_realtoxicity) == 0:
        raise ValueError("No datasets loaded. Check file paths.")
    
    print("\nSTEP 2: Filtering Toxic Content")
    print("=" * 70)
    
    df_toxigen_filtered = filter_toxic_content(df_toxigen, 'toxigen')
    df_realtoxicity_filtered = filter_toxic_content(df_realtoxicity, 'realtoxicity')
    
    print("\nSTEP 3: Combining Datasets")
    print("=" * 70)
    
    df_kb = combine_datasets(df_toxigen_filtered, df_realtoxicity_filtered)
    
    print("\nSTEP 4: Computing Statistics")
    print("=" * 70)
    
    stats = compute_statistics(df_kb)
    
    print("\nKnowledge Base Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  ToxiGen: {stats['toxigen_count']}")
    print(f"  RealToxicity: {stats['realtoxicity_count']}")
    print(f"  Mean toxicity: {stats['mean_toxicity']:.4f}")
    print(f"  Median toxicity: {stats['median_toxicity']:.4f}")
    
    print("\nSTEP 5: Saving Outputs")
    print("=" * 70)
    
    # Save KB CSV
    df_kb.to_csv(OUTPUT_CSV, index=False)
    print(f"Knowledge base saved to {OUTPUT_CSV}")
    
    # Save statistics
    save_statistics(stats, OUTPUT_STATS)
    
    # Populate ChromaDB
    populate_chromadb(df_kb)
    
    print("\n" + "=" * 70)
    print("Toxic KB construction complete!")
    print(f"Raw data: {OUTPUT_CSV}")
    print(f"Statistics: {OUTPUT_STATS}")
    print(f"ChromaDB: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print("=" * 70)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    build_toxic_kb()
