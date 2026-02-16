"""
Neutral Knowledge Base Construction

Scrapes Wikipedia articles on social topics to build a neutral knowledge base.
Creates both raw text corpus and ChromaDB collection.

Output:
    - neutral_kb_raw.csv: Raw Wikipedia articles
    - ChromaDB collection in chroma_neutralkb/

"""

import os
import re
import csv
import time
import requests
import uuid
from typing import List
from bs4 import BeautifulSoup
from tqdm import tqdm
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from textwrap import wrap

# ==================== CONFIGURATION ====================
WIKI_TOPICS: List[str] = [
    "Gender identity", "Transgender rights", "Feminism", "Sexual orientation",
    "Discrimination", "Race", "Racism", "Systemic racism", "Ethnic minority",
    "Colorism", "Religion", "Religious freedom", "Islamophobia",
    "Christianity and social issues", "Atheism", "Refugees", "Immigration law",
    "Refugee rights", "Asylum seekers", "LGBT rights", "Gay rights",
    "Bisexuality", "Non-binary gender", "Ableism", "Disability rights",
    "Neurodiversity", "Social exclusion", "Inclusion", "Human rights",
    "Civil rights", "Minority stress", "Discrimination law", "Migrant worker",
    "Cultural identity", "Nationality law", "Xenophobia", "Hate speech",
    "Freedom of expression", "Identity politics", "Multiculturalism",
    "Intercultural relations", "International human rights law",
    "Social justice", "Marginalization", "Women in politics", "LGBT in media",
    "Ethnic conflict", "Digital discrimination", "Transgender health",
    "Gender bias", "Educational inequality", "Affirmative action",
    "Gender pay gap"
]

HEADERS = {
    "User-Agent": "LLM-Toxicity-Benchmark (research@university.edu)"
}

# Output files
OUTPUT_CSV = "neutral_kb_raw.csv"
OUTPUT_JSONL = "neutral_kb_raw.jsonl"
CHROMA_DIR = "chroma_neutralkb"
COLLECTION_NAME = "neutralkb_knowledgebase"

# ChromaDB settings
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512

# ==================== WIKIPEDIA SCRAPER ====================
def get_full_wikipedia_text(title: str) -> str:
    """
    Fetch full article text from Wikipedia.
    
    Args:
        title: Wikipedia article title
        
    Returns:
        Cleaned article text
    """
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Extract paragraphs from main content
        paragraphs = [
            p.get_text().strip()
            for p in soup.select("div.mw-parser-output > p")
            if p.get_text().strip() and not p.get_text().lower().startswith(("this article", "see also"))
        ]
        
        return clean_text(" ".join(paragraphs))
    except Exception as e:
        print(f"Failed to fetch '{title}': {e}")
        return ""

def clean_text(text: str) -> str:
    """Remove citations and normalize whitespace."""
    text = re.sub(r"\[[^\]]*\]", "", text)  # Remove citations
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip()

# ==================== SAVE FUNCTIONS ====================
def save_to_csv(data: List[dict], filename: str) -> None:
    """Save topic-content pairs to CSV."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "topic", "text"])
        writer.writeheader()
        writer.writerows(data)

# ==================== CHROMADB POPULATION ====================
def populate_chromadb(data: List[dict]) -> None:
    """
    Populate ChromaDB with Wikipedia content.
    
    Args:
        data: List of topic-content dictionaries
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
    
    print("Populating ChromaDB with Wikipedia content...")
    
    for entry in tqdm(data, desc="Indexing to ChromaDB"):
        text = entry['text']
        
        # Split into chunks
        chunks = wrap(text, width=CHUNK_SIZE)
        
        # Generate unique IDs
        ids = [f"neutral_{uuid.uuid4()}_{i}" for i in range(len(chunks))]
        
        # Compute embeddings
        embeddings = embedder.encode(chunks)
        
        # Add to ChromaDB
        collection.add(
            documents=chunks,
            ids=ids,
            embeddings=embeddings,
            metadatas=[{"topic": entry['topic'], "id": entry['id']} for _ in chunks]
        )
    
    # Persist to disk
    chroma_client.persist()
    
    print(f"ChromaDB populated with {collection.count()} chunks")
    print(f"ChromaDB saved to: {CHROMA_DIR}")

# ==================== MAIN ROUTINE ====================
def build_neutral_kb() -> None:
    """Main function to build neutral KB from Wikipedia."""
    print("Starting Neutral KB Construction")
    print("=" * 70)
    print(f"Scraping {len(WIKI_TOPICS)} Wikipedia topics")
    print("=" * 70)
    
    topic_data = []
    
    for idx, topic in enumerate(tqdm(WIKI_TOPICS, desc="Scraping Wikipedia"), start=1):
        content = get_full_wikipedia_text(topic)
        
        if content:
            topic_data.append({
                "id": f"NeutralKB_{idx}",
                "topic": topic,
                "text": content
            })
        
        time.sleep(1.0)  # Be polite to Wikipedia
    
    print(f"\nSuccessfully scraped {len(topic_data)} topics")
    
    # Save to CSV
    print(f"\nSaving raw data to {OUTPUT_CSV}...")
    save_to_csv(topic_data, OUTPUT_CSV)
    print(f"Saved {len(topic_data)} entries to {OUTPUT_CSV}")
    
    # Populate ChromaDB
    populate_chromadb(topic_data)
    
    print("\n" + "=" * 70)
    print("Neutral KB construction complete!")
    print(f"Raw data: {OUTPUT_CSV}")
    print(f"ChromaDB: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print("=" * 70)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    build_neutral_kb()
