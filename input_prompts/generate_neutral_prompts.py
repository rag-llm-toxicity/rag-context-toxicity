"""
Neutral Prompt Generation

Generates Q&A pairs from the neutral knowledge base (Wikipedia articles).

Input:
    - neutral_kb_raw.csv: Raw Wikipedia content (from kb_construction)

Output:
    - neutral_prompts.csv: Question-answer pairs

Environment Variables Required:
    GROQ_API_KEY
"""

import os
import re
import json
import time
import csv
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIGURATION ====================
INPUT_CSV = "neutral_kb_raw.csv"
OUTPUT_CSV = "neutral_prompts.csv"

GROQ_MODEL = "llama3-70b-8192"
MAX_PAIRS_PER_CHUNK = 60
TEMPERATURE = 0.2
DESIRED_TOTAL_PAIRS = 10000

# ==================== GROQ CLIENT ====================
def get_groq_client():
    """Initialize Groq client from environment variable."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment. "
            "Set it in .env file or environment variables."
        )
    return Groq(api_key=api_key)

client = get_groq_client()

# ==================== TEXT PROCESSING ====================
def chunk_text(text: str, max_len: int = 2000) -> List[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.?!])\s+", text)
    chunks, current = [], ""
    
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_len:
            current += sent + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sent + " "
    
    if current:
        chunks.append(current.strip())
    
    return chunks

# ==================== Q&A GENERATION ====================
def parse_json_block(text: str) -> List[Dict[str, str]]:
    """Parse JSON array from LLM response."""
    try:
        text = text.strip()
        
        # Remove markdown
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE).strip()
        
        # Find JSON array
        json_match = re.search(r"\[\s*{.*?}\s*]", text, flags=re.DOTALL)
        if not json_match:
            raise ValueError("No JSON array found in LLM response.")
        
        return json.loads(json_match.group(0))
    except Exception as e:
        print("Failed to parse JSON:")
        print(text[:1000])
        raise e

def request_qa_pairs(chunk: str, n_pairs: int) -> List[Dict[str, str]]:
    """Request Q&A pairs from Groq API for a text chunk."""
    system_msg = (
        "You are an expert knowledge base generator.\n"
        "Your output MUST be a pure JSON array of objects (without any commentary, markdown, or explanation).\n"
        "Only include this JSON array. Do NOT include markdown formatting like ```json.\n"
        "Ground every answer strictly in the supplied text."
    )
    
    user_msg = (
        f"Generate {n_pairs} question-answer pairs in JSON array format "
        f"(keys: question, answer) based only on the text below.\n\n"
        f"<text>\n{chunk}\n</text>"
    )

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=4096
    )

    raw_output = resp.choices[0].message.content
    return parse_json_block(raw_output)

def generate_qa_from_text(text: str, target_pairs: int) -> List[Dict[str, str]]:
    """Generate Q&A pairs from article text."""
    qa_pairs: List[Dict[str, str]] = []
    pairs_needed = target_pairs
    
    for chunk in chunk_text(text, 2000):
        if pairs_needed <= 0:
            break
        
        num = min(pairs_needed, MAX_PAIRS_PER_CHUNK)
        
        try:
            qa_pairs.extend(request_qa_pairs(chunk, num))
            pairs_needed -= num
            time.sleep(0.5)
        except Exception as e:
            print(f"Groq error: {e}")
    
    return qa_pairs

# ==================== MAIN ROUTINE ====================
def generate_neutral_prompts():
    """Main function to generate neutral prompts."""
    print("Starting Neutral Prompt Generation")
    print("=" * 70)
    
    # Load KB
    print(f"Loading knowledge base from {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}\n"
            "Please run kb_construction/build_neutral_kb.py first"
        )
    
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} KB entries")
    
    # Calculate pairs per entry
    pairs_per_entry = DESIRED_TOTAL_PAIRS // len(df) + 1
    
    # Generate Q&A pairs
    all_pairs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Q&A"):
        if len(all_pairs) >= DESIRED_TOTAL_PAIRS:
            break
        
        text = row['text']
        pairs = generate_qa_from_text(text, pairs_per_entry)
        all_pairs.extend(pairs)
        
        time.sleep(1.0)
    
    # Trim to target
    if len(all_pairs) > DESIRED_TOTAL_PAIRS:
        all_pairs = all_pairs[:DESIRED_TOTAL_PAIRS]
    
    # Save to CSV
    print(f"\nSaving {len(all_pairs)} Q&A pairs to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'answer'])
        writer.writeheader()
        writer.writerows(all_pairs)
    
    print("=" * 70)
    print("Neutral prompt generation complete!")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Total pairs: {len(all_pairs)}")
    print("=" * 70)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    generate_neutral_prompts()
