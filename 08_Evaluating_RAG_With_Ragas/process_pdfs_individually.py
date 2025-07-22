import os
import glob
from semantic_chunk_tool import extract_and_chunk_paragraphs, token_count

def process_pdfs_from_directory(data_path="data"):
    """
    Load all PDF files from the specified path and process them one at a time
    """
    # Get all PDF files in the directory
    pdf_pattern = os.path.join(data_path, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in {data_path}")
        return {}
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    all_documents = {}
    
    # Process each PDF file individually
    for pdf_file in pdf_files:
        print(f"\nProcessing: {os.path.basename(pdf_file)}")
        print("-" * 50)
        
        try:
            # Extract and chunk paragraphs from this PDF
            paragraphs = extract_and_chunk_paragraphs(pdf_file)
            
            if paragraphs:
                # Store the results
                file_name = os.path.basename(pdf_file)
                all_documents[file_name] = paragraphs
                
                # Print summary for this file
                print(f"✓ Extracted {len(paragraphs)} paragraphs from {file_name}")
                
                # Show first paragraph as example
                if paragraphs:
                    print(f"First paragraph preview: {paragraphs[0][:200]}...")
                    print(f"Token count: {token_count(paragraphs[0])}")
                
            else:
                print(f"✗ No paragraphs extracted from {pdf_file}")
                
        except Exception as e:
            print(f"✗ Error processing {pdf_file}: {str(e)}")
    
    return all_documents

if __name__ == "__main__":
    # Process all PDFs
    documents = process_pdfs_from_directory()
    
    # Print overall summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    total_paragraphs = 0
    for file_name, paragraphs in documents.items():
        count = len(paragraphs)
        total_paragraphs += count
        print(f"{file_name}: {count} paragraphs")
    
    print(f"\nTotal paragraphs extracted: {total_paragraphs}")
    print(f"Total files processed: {len(documents)}") 