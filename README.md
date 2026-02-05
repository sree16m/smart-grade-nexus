# SmartGrade Nexus

The "Zero-Cost" Intelligence Engine for Question Paper Categorization and Answer Sheet Evaluation.

## 🛠️ Tech Stack
-   **Langauge**: Python 3.10+
-   **API Framework**: FastAPI
-   **LLM**: Google Gemini 1.5 Pro (via Google AI Studio Free Tier)
-   **Database**: Supabase (Free Tier - Postgres + pgvector)
-   **Parser**: PyMuPDF (fitz)

## 🚀 Setup Instructions

### 1. Prerequisites
-   Python 3.10 installed.
-   A **Google AI Studio API Key** (Free) -> [Get it here](https://aistudio.google.com/).
-   A **Supabase Project** (Free) -> [Create here](https://supabase.com/).

### 2. Environment Setup
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Edit `.env` and add your keys:
```ini
GOOGLE_API_KEY=AIzaSy...
SUPABASE_URL=https://xyz.supabase.co
SUPABASE_KEY=eyJ...
```

### 3. Database Setup (Supabase)
Run this SQL in your Supabase SQL Editor to enable vectors:
```sql
-- Enable Vector Extension
create extension if not exists vector;

-- Create Documents Table
create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(768) -- Matches Gemini 1.5 embedding size
);

-- Create Search Function (RPC)
create or replace function match_documents (
  query_embedding vector(768),
  match_threshold float,
  match_count int,
  filter jsonb
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  and documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

### 4. Install & Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 📚 API Endpoints
-   **Ingest Knowledge**: `POST /api/v1/knowledge/ingest`
-   **Categorize Questions**: `POST /api/v1/intelligence/categorize`
-   **Evaluate Answers**: `POST /api/v1/intelligence/evaluate`

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.
