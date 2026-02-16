"""
LLM Toxicity Evaluation using Groq API (Llama 3.1)

Generates responses using both prompt-only and RAG approaches, then scores with Detoxify.

Environment Variables Required:
    GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3

"""

import os
import pandas as pd
from detoxify import Detoxify
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
# Load API keys from environment
def load_groq_keys():
    """Load Groq API keys from environment variables."""
    keys = []
    i = 1
    while True:
        key = os.getenv(f'GROQ_API_KEY_{i}')
        if key is None:
            break
        keys.append(key)
        i += 1
    
    if not keys:
        raise ValueError(
            "No Groq API keys found. Set environment variables: "
            "GROQ_API_KEY_1, GROQ_API_KEY_2, etc."
        )
    
    return keys

GROQ_API_KEYS = load_groq_keys()

# Model and file paths
MODEL_NAME = "llama-3.1-8b-instant"
INPUT_CSV_PATH = "Combined_10K_Sample__RealToxicity___ToxiGen_.csv"
OUTPUT_CSV_PATH = "llama_toxic_toxicity_rag_vs_prompt_only.csv"

# ChromaDB configuration
CHROMA_DIR = "chroma_combined10k"
COLLECTION_NAME = "combined10k_knowledgebase"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Processing parameters
MAX_PROMPTS = 10000
SAVE_EVERY = 10

# ==================== API KEY MANAGER ====================
class GroqManager:
    """
    Manages multiple Groq API keys with automatic failover.
    
    When a key hits rate limits or fails, automatically switches to the next
    available key. This ensures uninterrupted processing of large datasets.
    """
    
    def __init__(self, keys):
        """
        Initialize the manager with a list of API keys.
        
        Args:
            keys (list): List of Groq API key strings
        """
        self.keys = keys
        self.index = 0
        self.client = self._get_client()

    def _get_client(self):
        """Create a new Groq client with the current API key."""
        return Groq(api_key=self.keys[self.index])

    def switch_key(self):
        """
        Switch to the next available API key.
        
        Returns:
            bool: True if switched successfully, False if no keys remain
        """
        if self.index + 1 < len(self.keys):
            self.index += 1
            print(f"Switching to backup Groq key #{self.index + 1}")
            self.client = self._get_client()
            return True
        else:
            print("All Groq API keys exhausted.")
            return False

    def generate(self, messages):
        """
        Generate text using the current API key, with automatic failover.
        
        Args:
            messages (list): List of message dicts for the chat completion
            
        Returns:
            Groq completion response object
            
        Raises:
            RuntimeError: If all API keys fail
        """
        while True:
            try:
                return self.client.chat.completions.create(
                    messages=messages,
                    model=MODEL_NAME
                )
            except Exception as e:
                print(f"Generation failed with key #{self.index + 1}: {e}")
                if not self.switch_key():
                    raise RuntimeError("All API keys failed or rate-limited.")

# ==================== INITIALIZATION ====================
print("Initializing components...")

# Initialize API manager
groq_manager = GroqManager(GROQ_API_KEYS)

# Initialize Detoxify toxicity classifier
# Using "original" model trained on Jigsaw toxicity datasets
tox_model = Detoxify("original")

# Initialize embedding function for ChromaDB
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# Initialize ChromaDB persistent client
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, 
    embedding_function=embedding_fn
)

print(f"ChromaDB collection loaded with {collection.count()} documents")

# ==================== DATA LOADING ====================
print(f"Loading dataset from {INPUT_CSV_PATH}...")

# Load the combined dataset
df = pd.read_csv(INPUT_CSV_PATH)

# Extract unique prompts (remove duplicates and NaN values)
prompts = df['prompt'].dropna().unique().tolist()[:MAX_PROMPTS]
print(f"Loaded {len(prompts)} unique prompts")

# ==================== RESUME HANDLING ====================
# Check if we're resuming from a previous run
if os.path.exists(OUTPUT_CSV_PATH):
    df_existing = pd.read_csv(OUTPUT_CSV_PATH)
    completed_prompts = set(df_existing["prompt"])
    print(f"Resuming from {len(completed_prompts)} completed prompts.")
else:
    df_existing = pd.DataFrame()
    completed_prompts = set()

results = []

# ==================== MAIN PROCESSING LOOP ====================
print("\nStarting toxicity evaluation experiment...")
print("=" * 70)

for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts"), start=1):
    # Skip already processed prompts
    if prompt in completed_prompts:
        continue

    print(f"\n[{idx}] Processing: {prompt[:60]}...")

    # -------------------- RAG Context Retrieval --------------------
    # Retrieve relevant context from the knowledge base
    try:
        query = collection.query(query_texts=[prompt], n_results=3)
        context = "\n".join(query['documents'][0])
        
        # Create RAG-enhanced prompt with retrieved context
        rag_prompt = (
            f"Use only the following context to answer the question or continue the statement:\n"
            f"{context}\n\n"
            f"Question: {prompt}"
        )
    except Exception as e:
        # Fallback to prompt-only if retrieval fails
        context = ""
        rag_prompt = prompt
        print(f"Retrieval failed: {e}")

    # -------------------- Prompt-Only Generation --------------------
    # Generate response without RAG (baseline condition)
    try:
        response_po = groq_manager.generate(
            messages=[{"role": "user", "content": prompt}]
        )
        text_po = response_po.choices[0].message.content.strip()
        
        # Score toxicity of the generated text
        tox_po = tox_model.predict(text_po)
        
    except Exception as e:
        text_po = ""
        tox_po = {}
        print(f"Prompt-only generation failed: {e}")

    # -------------------- RAG-Enhanced Generation --------------------
    # Generate response with RAG context (experimental condition)
    try:
        response_rag = groq_manager.generate(
            messages=[{"role": "user", "content": rag_prompt}]
        )
        text_rag = response_rag.choices[0].message.content.strip()
        
        # Score toxicity of the RAG-generated text
        tox_rag = tox_model.predict(text_rag)
        
    except Exception as e:
        text_rag = ""
        tox_rag = {}
        print(f"RAG generation failed: {e}")

    # -------------------- Store Results --------------------
    result = {
        "prompt": prompt,
        "prompt_only_response": text_po,
        "toxicity_prompt_only": tox_po.get("toxicity", ""),
        "rag_response": text_rag,
        "toxicity_rag": tox_rag.get("toxicity", "")
    }

    results.append(result)

    # -------------------- Periodic Checkpoint Save --------------------
    # Save progress every SAVE_EVERY prompts to prevent data loss
    if len(results) >= SAVE_EVERY:
        df_batch = pd.DataFrame(results)
        df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
        df_combined.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Saved {len(results)} new entries to {OUTPUT_CSV_PATH}")
        
        # Update tracking variables
        df_existing = df_combined
        completed_prompts.update(df_batch["prompt"])
        results = []

# ==================== FINAL SAVE ====================
# Save any remaining results
if results:
    df_batch = pd.DataFrame(results)
    df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
    df_combined.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nFinal save complete. Total prompts processed: {len(df_combined)}")

print("\n" + "=" * 70)
print("Experiment completed successfully!")
print(f"Results saved to: {OUTPUT_CSV_PATH}")
