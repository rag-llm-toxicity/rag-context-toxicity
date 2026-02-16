"""
Mixed Prompt Generation

Generates Q&A pairs from the mixed knowledge base (Reddit content).

Input:
    - mixed_kb_raw.csv: Raw Reddit content (from kb_construction)

Output:
    - mixed_prompts.csv: Question-answer pairs

Environment Variables Required:
    GROQ_API_KEY
"""

import os
import re
import time
import csv
import pandas as pd
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIGURATION ====================
INPUT_CSV = "mixed_kb_raw.csv"
OUTPUT_CSV = "mixed_prompts.csv"
INTERMEDIATE_DIR = "mixed_qna_blocks"

GROQ_MODEL = "llama-3.1-8b-instant"
QNA_PER_KB = 10
TARGET_TOTAL_PAIRS = 10000

os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

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

# ==================== PROMPT BUILDING ====================
def build_prompt(kb_text: str) -> str:
    """Build prompt for Q&A generation."""
    return f"""You are a QnA generator.

Given the following Reddit knowledge base content, generate 10 question-answer pairs that a user might ask based on the content.

Content:
\"\"\"{kb_text}\"\"\"

Return your output in the following format:

Question 1: ...
Answer 1: ...

Question 2: ...
Answer 2: ...

...
Question 10: ...
Answer 10: ...
"""

# ==================== Q&A GENERATION ====================
def generate_qna_block(kb_text: str) -> str:
    """Generate Q&A pairs from knowledge base text using Groq."""
    prompt = build_prompt(kb_text)
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8192,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error: {e}")
        return None

# ==================== Q&A EXTRACTION ====================
def extract_qna_pairs(qna_block: str) -> list:
    """Extract individual Q&A pairs from generated block."""
    pattern = re.compile(
        r'Question\d+:\s*(.*?)\s*Answer\d+:\s*(.*?)(?=Question\d+:|$)', 
        re.DOTALL
    )
    return [(q.strip(), a.strip()) for q, a in pattern.findall(qna_block)]

# ==================== MAIN ROUTINE ====================
def generate_mixed_prompts():
    """Main function to generate mixed prompts."""
    print("Starting Mixed Prompt Generation")
    print("=" * 70)
    
    # Load KB
    print(f"Loading knowledge base from {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}\n"
            "Please run kb_construction/build_mixed_kb.py first"
        )
    
    df = pd.read_csv(INPUT_CSV)
    
    # Remove stats rows
    df = df[~df['id'].str.contains('toxicity_score', na=False)]
    
    print(f"Loaded {len(df)} KB entries")
    
    # Step 1: Generate Q&A blocks
    print("\nStep 1: Generating Q&A blocks...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Q&A"):
        kb_id = row['id']
        kb_text = str(row['text'])[:4000]  # Truncate to reasonable length
        
        qna_block = generate_qna_block(kb_text)
        
        if qna_block:
            results.append({
                "id": kb_id,
                "qna_block": qna_block
            })
        
        time.sleep(1)  # Rate limiting
    
    # Save intermediate results
    intermediate_file = os.path.join(INTERMEDIATE_DIR, "qna_blocks.csv")
    pd.DataFrame(results).to_csv(intermediate_file, index=False)
    print(f"Saved Q&A blocks to {intermediate_file}")
    
    # Step 2: Extract individual pairs
    print("\nStep 2: Extracting individual Q&A pairs...")
    output_rows = []
    
    for result in tqdm(results, desc="Extracting pairs"):
        kb_id = result['id']
        qna_block = result['qna_block']
        
        for question, answer in extract_qna_pairs(qna_block):
            output_rows.append({
                'id': kb_id,
                'question': question,
                'answer': answer
            })
    
    # Create DataFrame and trim
    df_output = pd.DataFrame(output_rows)
    
    if len(df_output) > TARGET_TOTAL_PAIRS:
        df_output = df_output.head(TARGET_TOTAL_PAIRS)
    
    # Save final output
    df_output.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "=" * 70)
    print("Mixed prompt generation complete!")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Total pairs: {len(df_output)}")
    print("=" * 70)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    generate_mixed_prompts()
