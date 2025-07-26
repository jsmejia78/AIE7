# dataloader
from langchain_community.document_loaders.csv_loader import CSVLoader
# ===============================
# Load the data from the CSV file + Extract the consumer complaint narrative from the metadata
# ===============================

def load_data(num_docs: int = 50):
    loader = CSVLoader(
        file_path=f"./data/complaints.csv",
        metadata_columns=[
        "Date received", 
        "Product", 
        "Sub-product", 
        "Issue", 
        "Sub-issue", 
        "Consumer complaint narrative", 
        "Company public response", 
        "Company", 
        "State", 
        "ZIP code", 
        "Tags", 
        "Consumer consent provided?", 
        "Submitted via", 
        "Date sent to company", 
        "Company response to consumer", 
        "Timely response?", 
        "Consumer disputed?", 
        "Complaint ID"
        ]
    )

    # Load the data from the CSV file
    loan_complaint_data = loader.load()

    # Extract the consumer complaint narrative from the metadata
    for doc in loan_complaint_data:
        doc.page_content = doc.metadata["Consumer complaint narrative"]

    # Reduce the data to 50 documents for testing purposes - Golden Dataset!
    loan_complaint_data = loan_complaint_data[:num_docs]

    return loan_complaint_data