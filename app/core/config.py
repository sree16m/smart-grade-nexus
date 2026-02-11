import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "SmartGrade Nexus"
    API_V1_STR: str = "/api/v1"
    
    # Google AI Studio (Gemini)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Supabase (Vector DB + Storage)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    # AI Models
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    GENERATIVE_MODEL: str = "gemini-2.0-flash"
    
    # RAG Settings
    EMBEDDING_DIM: int = 768
    MATCH_THRESHOLD: float = 0.3
    MATCH_COUNT: int = 15
    DEFAULT_RAG_LIMIT: int = 5
    
    # Ingestion Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    INGESTION_BATCH_SIZE: int = 5
    MAX_EMBEDDING_CONCURRENCY: int = 10
    
    # Academic Defaults
    DEFAULT_BOARD: str = "SCERT"
    DEFAULT_CLASS: str = "9"
    CORRECTNESS_THRESHOLD: float = 0.9
    
    SUBJECT_MAP: Dict[str, List[str]] = {
        "mathematics": ["Maths", "Math", "Mathematics"],
        "maths": ["Mathematics", "Math", "Maths"],
        "math": ["Mathematics", "Maths", "Math"],
        "physics": ["Physics", "Physic"],
        "biology": ["Biology", "Bio"],
        "chemistry": ["Chemistry", "Chem"],
        "euclid geometry": ["Maths", "Mathematics", "Math"]
    }
    
    # OCR Fallback Thresholds
    OCR_MIN_TEXT_LENGTH: int = 50
    OCR_MIN_ENGLISH_DENSITY: float = 0.3
    OCR_MAX_GARBAGE_RATIO: float = 0.05

    # Subject-Specific Grading Rules
    # These are added to the general grading guidelines in agents.py
    SUBJECT_RULES: Dict[str, str] = {
        "Maths": "Prioritize step-by-step logical reasoning, correct use of LaTeX formulas, and numerical accuracy.",
        "Science": "Focus on precise scientific terminology, correct explanation of processes, and accurate reference to data/diagrams.",
        "Social Studies": "Prioritize historical accuracy, depth of concept explanation, and mention of key dates/events.",
        "Biology": "Check for correct biological terms, structural accuracy in descriptions, and process flow.",
        "English": "Focus on grammatical correctness, contextually appropriate vocabulary, and clarity of expression."
    }

    class Config:
        case_sensitive = True

settings = Settings()
