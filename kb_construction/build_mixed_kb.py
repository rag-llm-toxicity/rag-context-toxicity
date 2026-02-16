"""
Mixed Knowledge Base Construction

Scrapes Reddit posts and comments to build a mixed toxicity knowledge base.
Creates both raw text corpus and ChromaDB collection.

Output:
    - mixed_kb_raw.csv: Raw Reddit content
    - ChromaDB collection in chroma_mixedkb/

Environment Variables Required:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT

"""

import os
import csv
import uuid
import praw
import statistics
from detoxify import Detoxify
from tqdm import tqdm
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from textwrap import wrap
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIGURATION ====================
# Subreddits to scrape
SUBREDDITS = [
    "AmItheAsshole",
    "unpopularopinion", 
    "relationship_advice",
    "TrueOffMyChest"
]

# Scraping parameters
POST_LIMIT = 200  # Posts per subreddit
MAX_COMMENTS = 200  # Comments per post

# Output files
OUTPUT_CSV = "mixed_kb_raw.csv"
CHROMA_DIR = "chroma_mixedkb"
COLLECTION_NAME = "mixedkb_knowledgebase"

# ChromaDB settings
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512

# ==================== REDDIT CLIENT ====================
def get_reddit_client():
    """Initialize Reddit client from environment variables."""
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'ToxicKB/1.0')
    
    if not client_id or not client_secret:
        raise ValueError(
            "Reddit credentials not found. Set REDDIT_CLIENT_ID and "
            "REDDIT_CLIENT_SECRET in .env file."
        )
    
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

# ==================== REDDIT SCRAPER ====================
def scrape_subreddit(subreddit_name: str, reddit_client) -> list:
    """
    Scrape posts and comments from a subreddit.
    
    Args:
        subreddit_name: Name of subreddit
        reddit_client: PRAW Reddit instance
        
    Returns:
        List of knowledge base entries
    """
    print(f"\nScraping subreddit: r/{subreddit_name}")
    
    subreddit = reddit_client.subreddit(subreddit_name)
    knowledge_base_rows = []
    kb_counter = 1

    for post in tqdm(subreddit.hot(limit=POST_LIMIT), desc=f"r/{subreddit_name}"):
        # Combine post title and body
        combined_text = post.title + "\n" + post.selftext

        # Get comments
        post.comments.replace_more(limit=0)
        comments = post.comments.list()
        selected_comments = comments[:MAX_COMMENTS]

        for comment in selected_comments:
            combined_text += "\n" + comment.body

        # Save entry
        knowledge_base_rows.append({
            'id': f'MixedKB_{subreddit_name}_{kb_counter}',
            'subreddit': subreddit_name,
            'text': combined_text.replace('\n', ' ').strip()
        })
        kb_counter += 1

    print(f"Scraped {len(knowledge_base_rows)} entries from r/{subreddit_name}")
    
    return knowledge_base_rows

# ==================== TOXICITY ANALYSIS ====================
def analyze_toxicity(entries: list) -> dict:
    """
    Analyze toxicity of KB entries.
    
    Args:
        entries: List of KB entries
        
    Returns:
        Dictionary of toxicity statistics
    """
    print("\nAnalyzing toxicity scores...")
    
    tox_model = Detoxify('original')
    toxicity_scores = []
    
    for entry in tqdm(entries, desc="Scoring toxicity"):
        try:
            score = tox_model.predict(entry['text'])['toxicity']
            entry['toxicity_score'] = score
            toxicity_scores.append(score)
        except:
            entry['toxicity_score'] = 0.0
            toxicity_scores.append(0.0)
    
    # Compute statistics
    if len(toxicity_scores) > 0:
        stats = {
            'median': statistics.median(toxicity_scores),
            'mean': statistics.mean(toxicity_scores),
            'stdev': statistics.stdev(toxicity_scores) if len(toxicity_scores) > 1 else 0.0,
            'min': min(toxicity_scores),
            'max': max(toxicity_scores)
        }
    else:
        stats = {'median': 0.0, 'mean': 0.0, 'stdev': 0.0, 'min': 0.0, 'max': 0.0}
    
    return stats

# ==================== SAVE FUNCTIONS ====================
def save_to_csv(entries: list, filename: str, stats: dict) -> None:
    """Save KB entries and statistics to CSV."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'subreddit', 'text', 'toxicity_score'])
        writer.writeheader()
        writer.writerows(entries)
        
        # Write stats at the end
        writer.writerow({
            'id': 'median_toxicity_score',
            'subreddit': '',
            'text': '',
            'toxicity_score': stats['median']
        })
        writer.writerow({
            'id': 'average_toxicity_score',
            'subreddit': '',
            'text': '',
            'toxicity_score': stats['mean']
        })
        writer.writerow({
            'id': 'stdev_toxicity_score',
            'subreddit': '',
            'text': '',
            'toxicity_score': stats['stdev']
        })

# ==================== CHROMADB POPULATION ====================
def populate_chromadb(entries: list) -> None:
    """
    Populate ChromaDB with Reddit content.
    
    Args:
        entries: List of KB entries
    """
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
    
    print("Populating ChromaDB with Reddit content...")
    
    for entry in tqdm(entries, desc="Indexing to ChromaDB"):
        text = entry['text']
        
        # Split into chunks
        chunks = wrap(text, width=CHUNK_SIZE)
        
        # Generate unique IDs
        ids = [f"mixed_{uuid.uuid4()}_{i}" for i in range(len(chunks))]
        
        # Compute embeddings
        embeddings = embedder.encode(chunks)
        
        # Add to ChromaDB
        collection.add(
            documents=chunks,
            ids=ids,
            embeddings=embeddings,
            metadatas=[{
                "subreddit": entry['subreddit'],
                "id": entry['id'],
                "toxicity_score": entry.get('toxicity_score', 0.0)
            } for _ in chunks]
        )
    
    # Persist to disk
    chroma_client.persist()
    
    print(f"ChromaDB populated with {collection.count()} chunks")
    print(f"ChromaDB saved to: {CHROMA_DIR}")

# ==================== MAIN ROUTINE ====================
def build_mixed_kb():
    """Main function to build mixed KB from Reddit."""
    print("Starting Mixed KB Construction")
    print("=" * 70)
    print(f"Scraping {len(SUBREDDITS)} subreddits")
    print("=" * 70)
    
    # Initialize Reddit client
    reddit_client = get_reddit_client()
    
    # Scrape all subreddits
    all_entries = []
    
    for subreddit in SUBREDDITS:
        entries = scrape_subreddit(subreddit, reddit_client)
        all_entries.extend(entries)
    
    print(f"\nTotal entries scraped: {len(all_entries)}")
    
    # Analyze toxicity
    stats = analyze_toxicity(all_entries)
    
    print("\nToxicity Statistics:")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std Dev: {stats['stdev']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    
    # Save to CSV
    print(f"\nSaving to {OUTPUT_CSV}...")
    save_to_csv(all_entries, OUTPUT_CSV, stats)
    print(f"Saved {len(all_entries)} entries")
    
    # Populate ChromaDB
    populate_chromadb(all_entries)
    
    print("\n" + "=" * 70)
    print("Mixed KB construction complete!")
    print(f"Raw data: {OUTPUT_CSV}")
    print(f"ChromaDB: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print("=" * 70)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    build_mixed_kb()
