import os

from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, Docx2txtLoader, BSHTMLLoader

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = f"{ROOT_PATH}/DATA"
PERSIST_PATH = f"{ROOT_PATH}/STORAGE"
INGEST_FILE_PATH = f"{ROOT_PATH}/FEED/ingest.txt"
QUESTION_FILE_PATH = f"{ROOT_PATH}/FEED/questions.txt"

INGEST_THREADS = os.cpu_count() or 8

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".html": BSHTMLLoader
}

CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_PATH,
    anonymized_telemetry=False
)

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"