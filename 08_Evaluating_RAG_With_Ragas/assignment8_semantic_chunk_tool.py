from langchain_community.document_loaders import PyMuPDFLoader
import re
import os
import tiktoken

def extract_and_chunk_paragraphs(file_path):
    # Step 1: Open the PDF
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # Combine all pages into one string
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Step 3: Normalize line breaks
    # Replace single newlines (inside a paragraph) with space
    # But keep double newlines to mark paragraph boundaries
    normalized = re.sub(r'(?<!\n)\n(?!\n)', ' ', full_text)
    
    # Step 4: Split into paragraphs based on double line breaks
    paragraphs = [p.strip() for p in normalized.split('\n\n') if p.strip()]
    
    # Step 4: Merge chunks if the next "paragraph" starts with lowercase (likely same thought)
    merged_paragraphs = []
    buffer = ""

    for para in paragraphs:
        if para and para[0].islower():
            # Continuation → append to previous
            buffer += " " + para
        else:
            # New paragraph
            if buffer:
                merged_paragraphs.append(buffer.strip())
            buffer = para
    if buffer:
        merged_paragraphs.append(buffer.strip())

    return merged_paragraphs

def is_within_token_limit(text, max_tokens=8191, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(text))
    return token_count <= max_tokens

# Example usage:
pdf_file = os.getcwd() + "/data/The_Direct_Loan_Program.pdf"
paragraphs = extract_and_chunk_paragraphs(pdf_file)

print(f"Extracted {len(paragraphs)} paragraphs.")
print("-" * 100)
print("-" * 100)
print("-" * 100)

for i, paragraph in enumerate(paragraphs):
    if i < 3:
        print(f"{i}: {paragraph}")
        print(f"-------Token count: {len(tiktoken.encoding_for_model('text-embedding-3-small').encode(paragraph))}")
        print("-" * 100)
        print("-" * 100)
        print("-" * 100)
        print("-" * 100)
        print("-" * 100)