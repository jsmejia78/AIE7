from langchain_community.document_loaders import PyMuPDFLoader
import re
import tiktoken

def extract_and_chunk_paragraphs(file_path):
    # Step 1: Open the PDF
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    if not docs:
        print(f"No content extracted from {file_path}")
        return []

    # Step 2: Combine all pages into one string
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Step 3: Normalize line breaks
    # Replace single newlines (inside a paragraph) with space
    # But keep double newlines to mark paragraph boundaries
    normalized = re.sub(r'(?<!\n)\n(?!\n)', ' ', full_text)
    
    # Step 4: Split into paragraphs based on double line breaks
    paragraphs = [p.strip() for p in normalized.split('\n\n') if p.strip()]
    
    # Step 5: Merge chunks if the next "paragraph" likely continues previous
    merged_paragraphs = []
    buffer = ""
    for para in paragraphs:
        if para and re.match(r'^[a-z"\']', para):
            # Likely a continuation → append to previous
            buffer += " " + para
        else:
            # New paragraph
            if buffer:
                merged_paragraphs.append(buffer.strip())
            buffer = para
    if buffer:
        merged_paragraphs.append(buffer.strip())

    return merged_paragraphs

# Create a token counter once for efficiency
encoding = tiktoken.encoding_for_model("text-embedding-3-small")

def token_count(text):
    return len(encoding.encode(text))

def is_within_token_limit(text, max_tokens=8191):
    return token_count(text) <= max_tokens
