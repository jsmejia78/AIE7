from langchain_community.document_loaders import PyMuPDFLoader
import re
import tiktoken
import nltk

nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# GPT-4o tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")
MAX_TOKENS = 1000

def num_tokens(text):
    return len(tokenizer.encode(text))

def split_paragraph(paragraph, max_tokens=MAX_TOKENS):
    sentences = sent_tokenize(paragraph)
    chunks = []
    current_chunk = []
    current_tokens = 0

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = num_tokens(sent)

        if current_tokens + sent_tokens > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Add 1–2 sentence overlap
                overlap = sentences[i - 2:i] if i >= 2 else sentences[max(0, i - 1):i]
                current_chunk = list(overlap)
                current_tokens = sum(num_tokens(s) for s in current_chunk)
            else:
                # Single sentence too large (rare), force-split
                chunks.append(sent)
                current_chunk = []
                current_tokens = 0
        else:
            current_chunk.append(sent)
            current_tokens += sent_tokens
            i += 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def extract_and_chunk_paragraphs(file_path, split_if_exceeds_tokens_limit = False):
    # Step 1: Open the PDF
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        print(f"No content extracted from {file_path}")
        return []

    # Step 2: Combine all pages into one string
    full_text = "\n".join([doc.page_content for doc in docs])

    # Step 3: Normalize line breaks
    normalized = re.sub(r'(?<!\n)\n(?!\n)', ' ', full_text)

    # Step 4: Split into paragraphs
    paragraphs = [p.strip() for p in normalized.split('\n\n') if p.strip()]

    # Step 5: Merge likely continuation paragraphs
    merged_paragraphs = []
    buffer = ""
    for para in paragraphs:
        if para and re.match(r'^[a-z"\']', para):
            buffer += " " + para
        else:
            if buffer:
                merged_paragraphs.append(buffer.strip())
            buffer = para
    if buffer:
        merged_paragraphs.append(buffer.strip())

    # Step 6: Split long paragraphs by token length
    if split_if_exceeds_tokens_limit:
        final_chunks = []
        for para in merged_paragraphs:
            if num_tokens(para) > MAX_TOKENS:
                final_chunks.extend(split_paragraph(para, MAX_TOKENS))
            else:
                final_chunks.append(para)
    else:
        final_chunks = merged_paragraphs

    return final_chunks

def print_chunk_summary(chunks, max_chunks=8):
    """Print first N chunks with their token counts"""
    print(f"\n{'='*80}")
    print(f"CHUNK SUMMARY - Showing first {min(max_chunks, len(chunks))} chunks out of {len(chunks)} total")
    print(f"{'='*80}")
    
    for i, chunk in enumerate(chunks[:max_chunks]):
        token_count = num_tokens(chunk)
        print(f"\nChunk {i+1} ({token_count} tokens):")
        print("-" * 50)
        # Print first 200 characters of the chunk for preview
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(preview)
        print(f"{'='*80}")
    
    if len(chunks) > max_chunks:
        print(f"\n... and {len(chunks) - max_chunks} more chunks")


# Example usage:
'''
if __name__ == "__main__":
    import os
    
    pdf_file = os.path.join(os.getcwd(), "data", "The_Direct_Loan_Program.pdf")
    
    if os.path.exists(pdf_file):
        print(f"Processing: {pdf_file}")
        chunks = extract_and_chunk_paragraphs(pdf_file)
        
        # Print summary of first 8 chunks
        print_chunk_summary(chunks, max_chunks=8)
        
        # Print overall statistics
        total_tokens = sum(num_tokens(chunk) for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        max_chunk_tokens = max(num_tokens(chunk) for chunk in chunks) if chunks else 0
        min_chunk_tokens = min(num_tokens(chunk) for chunk in chunks) if chunks else 0
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per chunk: {avg_tokens:.1f}")
        print(f"Max chunk tokens: {max_chunk_tokens}")
        print(f"Min chunk tokens: {min_chunk_tokens}")
        print(f"Chunks over 1000 tokens: {sum(1 for chunk in chunks if num_tokens(chunk) > 1000)}")
    else:
        print(f"File not found: {pdf_file}")

'''