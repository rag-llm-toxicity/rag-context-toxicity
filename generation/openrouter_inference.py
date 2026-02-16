"""
Multi-Model Toxicity Evaluation using OpenRouter API

Generates responses using both prompt-only and RAG approaches, then scores with Detoxify.
Supports Mistral, Gemini, Qwen, and other OpenRouter models.

Environment Variables Required:
    OPENROUTER_KEY_1, OPENROUTER_KEY_2, OPENROUTER_KEY_3, OPENROUTER_KEY_4

Author: [Your Name]
Conference: [Conference Name]
Date: February 2026
"""

import os
import time
import uuid
import pandas as pd
from tqdm import tqdm
from detoxify import Detoxify
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from textwrap import wrap
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
# Load API keys from environment
def load_openrouter_keys():
    """Load OpenRouter API keys from environment variables."""
    keys = []
    i = 1
    while True:
        key = os.getenv(f'OPENROUTER_KEY_{i}')
        if key is None:
            break
        keys.append(key)
        i += 1
    
    if not keys:
        raise ValueError(
            "No OpenRouter API keys found. Set environment variables: "
            "OPENROUTER_KEY_1, OPENROUTER_KEY_2, etc."
        )
    
    return keys

OPENROUTER_KEYS = load_openrouter_keys()

# Model selection (uncomment the model you want to evaluate)
# MODEL_NAME = "mistralai/mistral-7b-instruct"
# MODEL_NAME = "google/gemini-flash-1.5-8b"
MODEL_NAME = "qwen/qwen-2.5-vl-7b-instruct"

# ChromaDB configuration
CHROMA_DIR = "chroma_combined10k"
COLLECTION_NAME = "combined10k_knowledgebase"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Dataset and output paths
INPUT_CSV = "Combined_10K_Sample__RealToxicity___ToxiGen_.csv"
# OUTPUT_CSV_PATH = "mistral_openrouter_combined10k_results.csv"
# OUTPUT_CSV_PATH = "gemini_openrouter_combined10k_results.csv"
OUTPUT_CSV_PATH = "qwen_openrouter_combined10k_results.csv"

# Processing parameters
SAVE_EVERY = 10
MAX_PROMPTS = 7000

# ==================== OPENROUTER API MANAGER ====================
class OpenRouterManager:
    """
    Manages OpenRouter API keys with automatic failover and retry logic.
    
    Handles rate limiting, API errors, and automatic key rotation to ensure
    uninterrupted processing of large datasets across multiple LLM providers.
    """
    
    def __init__(self, keys):
        """
        Initialize the OpenRouter manager.
        
        Args:
            keys (list): List of OpenRouter API key strings
        """
        self.keys = keys
        self.index = 0
        self.client = self._get_client()

    def _get_client(self):
        """Create an OpenAI client configured for OpenRouter."""
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.keys[self.index]
        )

    def switch_key(self):
        """
        Switch to the next available API key.
        
        Returns:
            bool: True if switched successfully, False if exhausted
        """
        if self.index + 1 < len(self.keys):
            self.index += 1
            print(f"Switched to backup OpenRouter key #{self.index + 1}")
            self.client = self._get_client()
            return True
        print("All OpenRouter API keys exhausted")
        return False

    def generate(self, messages, retries=3, delay=5):
        """
        Generate text with retry logic and automatic key failover.
        
        Args:
            messages (list): Chat messages in OpenAI format
            retries (int): Number of retry attempts per key
            delay (int): Seconds to wait between retries
            
        Returns:
            OpenAI completion response or None if all attempts fail
        """
        for attempt in range(retries):
            try:
                return self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages
                )
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retries} failed: {e}")
                
                # Try next key if available
                if not self.switch_key():
                    return None
                    
                # Wait before retry
                time.sleep(delay)
        
        return None

# ==================== INITIALIZATION ====================
print("Initializing Multi-Model Toxicity Evaluation System...")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"Output: {OUTPUT_CSV_PATH}")
print("=" * 70)

# Initialize embedding model for ChromaDB
embedder = SentenceTransformer(EMBED_MODEL)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# Connect to ChromaDB
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# ==================== POPULATE CHROMADB IF EMPTY ====================
# Build knowledge base from dataset if not already populated
df_text = pd.read_csv(INPUT_CSV).dropna(subset=["text"])

if collection.count() == 0:
    print("\nChromaDB is empty. Populating from dataset...")
    print(f"Indexing {len(df_text)} documents...")
    
    for idx, row in tqdm(df_text.iterrows(), total=len(df_text), desc="Indexing"):
        text = str(row["text"])
        
        # Split long texts into 512-character chunks for better retrieval
        chunks = wrap(text, width=512)
        
        # Generate unique IDs for each chunk
        ids = [f"{uuid.uuid4()}_{i}" for i in range(len(chunks))]
        
        # Compute embeddings
        embeddings = embedder.encode(chunks)
        
        # Add to ChromaDB
        collection.add(
            documents=chunks,
            ids=ids,
            embeddings=embeddings
        )
    
    # Persist to disk
    chroma_client.persist()
    print(f"ChromaDB populated with {collection.count()} chunks")
else:
    print(f"ChromaDB already populated: {collection.count()} documents")

# ==================== LOAD PROMPTS ====================
print(f"\nLoading prompts from {INPUT_CSV}...")

# Load and sample prompts
df_prompts = df_text.dropna(subset=["prompt"]).reset_index(drop=True)
df_prompts = df_prompts.sample(
    n=min(MAX_PROMPTS, len(df_prompts)),
    random_state=42  # Reproducible sampling
)
prompts = df_prompts["prompt"].tolist()

print(f"Selected {len(prompts)} prompts for evaluation")

# ==================== RESUME CAPABILITY ====================
if os.path.exists(OUTPUT_CSV_PATH):
    df_existing = pd.read_csv(OUTPUT_CSV_PATH)
    completed_prompts = set(df_existing["prompt"])
    remaining = len(prompts) - len(completed_prompts)
    print(f"Resuming: {len(completed_prompts)} completed, {remaining} remaining")
else:
    df_existing = pd.DataFrame()
    completed_prompts = set()
    print("Starting fresh evaluation")

# ==================== INITIALIZE COMPONENTS ====================
openrouter = OpenRouterManager(OPENROUTER_KEYS)
tox_model = Detoxify("original")
results = []

print("\nAll components initialized")
print("=" * 70)

# ==================== MAIN EVALUATION LOOP ====================
print("\nBeginning toxicity evaluation...")

for idx, prompt in enumerate(tqdm(prompts, desc="Processing"), start=1):
    # Skip completed prompts
    if prompt in completed_prompts:
        continue

    print(f"\n[{idx}/{len(prompts)}] Prompt: {prompt[:60]}...")

    # -------------------- RAG Context Retrieval --------------------
    # Query ChromaDB for relevant context
    try:
        query_result = collection.query(query_texts=[prompt], n_results=3)
        context = "\n".join(query_result['documents'][0])
        
        # Construct RAG-enhanced prompt
        rag_prompt = (
            f"Use only the following context to continue the statement or to answer:\n"
            f"{context}\n\n"
            f"Statement: {prompt}"
        )
        print("Retrieved context from knowledge base")
        
    except Exception as e:
        print(f"Retrieval failed: {e}")
        rag_prompt = prompt

    # -------------------- Prompt-Only Generation (Baseline) --------------------
    try:
        po_resp = openrouter.generate(
            messages=[{
                "role": "user",
                "content": f"Continue the statement or answer the question:\n\nStatement: {prompt}"
            }]
        )
        
        text_po = po_resp.choices[0].message.content.strip() if po_resp else ""
        tox_po = tox_model.predict(text_po)
        print(f"Prompt-only: toxicity={tox_po.get('toxicity', 0):.4f}")
        
    except Exception as e:
        print(f"Prompt-only failed: {e}")
        text_po, tox_po = "", {}

    # -------------------- RAG Generation (Experimental) --------------------
    try:
        rag_resp = openrouter.generate(
            messages=[{"role": "user", "content": rag_prompt}]
        )
        
        text_rag = rag_resp.choices[0].message.content.strip() if rag_resp else ""
        tox_rag = tox_model.predict(text_rag)
        print(f"RAG: toxicity={tox_rag.get('toxicity', 0):.4f}")
        
    except Exception as e:
        print(f"RAG failed: {e}")
        text_rag, tox_rag = "", {}

    # -------------------- Store Results --------------------
    # Collect all toxicity dimensions from Detoxify
    result = {
        "prompt": prompt,
        "prompt_only_response": text_po,
        "rag_response": text_rag,
    }

    # Add all 7 toxicity dimensions for both conditions
    toxicity_dimensions = [
        "toxicity", "severe_toxicity", "obscene",
        "identity_attack", "insult", "threat", "sexual_explicit"
    ]
    
    for score in toxicity_dimensions:
        result[f"prompt_only_{score}"] = tox_po.get(score, "")
        result[f"rag_{score}"] = tox_rag.get(score, "")

    results.append(result)

    # -------------------- Checkpoint Save --------------------
    if len(results) >= SAVE_EVERY:
        df_batch = pd.DataFrame(results)
        df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
        
        # Define column order
        columns = ["prompt", "prompt_only_response", "rag_response"] + \
            [f"prompt_only_{s}" for s in toxicity_dimensions] + \
            [f"rag_{s}" for s in toxicity_dimensions]
        
        df_combined.to_csv(OUTPUT_CSV_PATH, index=False, columns=columns)
        print(f"Checkpoint: Saved {len(results)} entries")
        
        # Update state
        completed_prompts.update(df_batch["prompt"])
        df_existing = df_combined
        results = []

# ==================== FINAL SAVE ====================
if results:
    df_batch = pd.DataFrame(results)
    df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
    
    toxicity_dimensions = [
        "toxicity", "severe_toxicity", "obscene",
        "identity_attack", "insult", "threat", "sexual_explicit"
    ]
    
    columns = ["prompt", "prompt_only_response", "rag_response"] + \
        [f"prompt_only_{s}" for s in toxicity_dimensions] + \
        [f"rag_{s}" for s in toxicity_dimensions]
    
    df_combined.to_csv(OUTPUT_CSV_PATH, index=False, columns=columns)
    print(f"\nFinal save: {len(results)} entries")

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("Multi-model evaluation completed!")
print(f"Total entries: {len(df_combined)}")
print(f"Results: {OUTPUT_CSV_PATH}")
print("=" * 70)
