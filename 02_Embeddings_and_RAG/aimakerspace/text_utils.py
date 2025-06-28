import os
from typing import List
import pymupdf


class PdfFileLoader:
    """
    A class for loading and extracting text content from PDF files.
    
    This loader can handle both single PDF files and directories containing PDF files.
    It uses PyMuPDF (fitz) to extract text content from PDF documents.
    
    Attributes:
        documents (list): A list to store extracted text content from PDF files
        path (str): The file path or directory path to load PDFs from
    """
    
    def __init__(self, path: str):
        """
        Initialize the PDF file loader.
        
        Args:
            path (str): Path to a PDF file or directory containing PDF files
        """
        self.documents = []
        self.path = path

    def load(self):
        """
        Load PDF content based on the provided path.
        
        If the path is a directory, loads all PDF files in the directory.
        If the path is a single PDF file, loads that file.
        
        Raises:
            ValueError: If the path is neither a valid directory nor a PDF file
        """
        if os.path.isdir(self.path):
            # Load all PDF files in the directory
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            # Load a single PDF file
            self.documents.append(self.load_file(self.path))
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self, file_path: str) -> str:
        """
        Extract text content from a single PDF file.
        
        Opens the PDF file using PyMuPDF and extracts text from all pages,
        concatenating them into a single string.
        
        Args:
            file_path (str): Path to the PDF file to load
            
        Returns:
            str: The extracted text content from the PDF file
        """
        # Open the PDF document
        doc = pymupdf.open(file_path)
        full_text = ""
        
        # Extract text from each page
        for page in doc:
            full_text += page.get_text()
            
        return full_text

    def load_directory(self):
        """
        Load all PDF files found in the specified directory.
        
        Recursively walks through the directory and its subdirectories,
        finding all files with .pdf extension and loading their content.
        """
        # Walk through the directory tree
        for root, _, files in os.walk(self.path):
            for file in files:
                # Check if the file is a PDF
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    # Load the PDF file and add its content to documents
                    self.documents.append(self.load_file(file_path))

    def load_documents(self) -> list:
        """
        Load all PDF documents and return the extracted text content.
        
        This is the main method to use for loading PDF content. It calls
        the load() method and returns the list of extracted text documents.
        
        Returns:
            list: A list of strings, where each string contains the text
                  content from one PDF file
        """
        self.load()
        return self.documents



class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":

    type_doc = "pdf"

    if type_doc == "pdf":
        loader = PdfFileLoader("data/US_economy.pdf")
        loader.load()
        splitter = CharacterTextSplitter()
        chunks = splitter.split_texts(loader.documents)
        print(len(chunks))
        print(chunks[0])
        print("--------")
        print(chunks[1])
        print("--------")
        print(chunks[-2])
        print("--------")
        print(chunks[-1])
    else:
        loader = TextFileLoader("data/KingLear.txt")
        loader.load()
        splitter = CharacterTextSplitter()
        chunks = splitter.split_texts(loader.documents)
        print(len(chunks))
        print(chunks[0])
        print("--------")
        print(chunks[1])
        print("--------")
        print(chunks[-2])
        print("--------")
        print(chunks[-1])
