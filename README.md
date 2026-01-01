# Jobzilla

Job search with semantic matching using vector embeddings. Finds jobs that match your resume using AI-powered similarity search.

## What It Does

- Fetches jobs from Adzuna and JSearch APIs
- Generates embeddings for jobs and resumes using Nomic AI
- Stores everything in Supabase with vector search (HNSW indexes)
- Matches resumes to jobs based on semantic similarity

## Setup

### 1. Modal (for running embedding scripts)

```bash
pip install modal
modal setup
```

Set secrets:
```bash
modal secret create adzuna-secret ADZUNA_APP_ID=xxx ADZUNA_APP_KEY=xxx
modal secret create rapidapi-secret RAPIDAPI_KEY=xxx
modal secret create supabase-secret SUPABASE_URL=xxx SUPABASE_KEY=xxx
```

### 2. Supabase MCP (for database operations)

Follow the [Supabase MCP guide](https://supabase.com/docs/guides/getting-started/mcp) to set up, then:

```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json with your project_ref
```

Apply the schema:
```bash
# Run schema.sql in your Supabase project
```

## Usage

Fetch and embed jobs:
```bash
modal run job_embeddings_adzuna.py
modal run job_embeddings_jsearch.py
```

Generate resume embeddings:
```bash
# Place your resume.json in data/
modal run resume_embeddings.py
```

Match jobs to your resume (via Supabase MCP):
```sql
SELECT * FROM match_jobs_to_resume(
  (SELECT embedding FROM resumes LIMIT 1),
  100
);
```
