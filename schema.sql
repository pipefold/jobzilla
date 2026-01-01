-- Jobzilla Database Schema
-- Requires pgvector extension for vector similarity search

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Jobs table: Stores job listings with embeddings
CREATE TABLE IF NOT EXISTS jobs (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    title TEXT,
    description TEXT,
    company TEXT,
    location TEXT,
    country TEXT,
    employment_type TEXT,
    posted_at TIMESTAMPTZ,
    apply_link TEXT,
    salary_min NUMERIC,
    salary_max NUMERIC,
    salary_is_predicted INTEGER,  -- 0 = actual, 1 = predicted
    source TEXT,  -- Source of job listing (e.g., 'JSearch', 'Adzuna', 'LinkedIn')
    embedding vector(768),  -- Nomic v1.5 produces 768-dimensional embeddings
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for vector similarity search on jobs
-- Using HNSW for better recall and performance
CREATE INDEX IF NOT EXISTS jobs_embedding_idx ON jobs
USING hnsw (embedding vector_cosine_ops);

-- Index for filtering jobs
CREATE INDEX IF NOT EXISTS jobs_job_id_idx ON jobs(job_id);
CREATE INDEX IF NOT EXISTS jobs_posted_at_idx ON jobs(posted_at);
CREATE INDEX IF NOT EXISTS jobs_source_idx ON jobs(source);
CREATE INDEX IF NOT EXISTS jobs_salary_max_idx ON jobs(salary_max DESC NULLS LAST);


-- Resumes table: Stores JSON Resume documents with embeddings
CREATE TABLE IF NOT EXISTS resumes (
    id BIGSERIAL PRIMARY KEY,
    resume_id TEXT UNIQUE NOT NULL,  -- Usually email or generated ID
    name TEXT,
    email TEXT,
    professional_title TEXT,
    json_data JSONB NOT NULL,  -- Full JSON Resume document
    embedding vector(768),  -- Same dimension as jobs for matching
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for vector similarity search on resumes
-- Using HNSW for better recall and performance
CREATE INDEX IF NOT EXISTS resumes_embedding_idx ON resumes
USING hnsw (embedding vector_cosine_ops);

-- Index for quick lookups
CREATE INDEX IF NOT EXISTS resumes_resume_id_idx ON resumes(resume_id);
CREATE INDEX IF NOT EXISTS resumes_email_idx ON resumes(email);

-- JSONB indexes for querying within resume data
CREATE INDEX IF NOT EXISTS resumes_json_data_idx ON resumes USING gin(json_data);


-- Function: Find jobs matching a resume
-- Returns top matching jobs based on embedding similarity
CREATE OR REPLACE FUNCTION match_jobs_to_resume(
    query_embedding vector(768),
    match_count int DEFAULT 10
)
RETURNS TABLE (
    job_id TEXT,
    title TEXT,
    company TEXT,
    location TEXT,
    description TEXT,
    apply_link TEXT,
    salary_min NUMERIC,
    salary_max NUMERIC,
    salary_is_predicted INTEGER,
    source TEXT,
    similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        jobs.job_id,
        jobs.title,
        jobs.company,
        jobs.location,
        jobs.description,
        jobs.apply_link,
        jobs.salary_min,
        jobs.salary_max,
        jobs.salary_is_predicted,
        jobs.source,
        1 - (jobs.embedding <=> query_embedding) as similarity
    FROM jobs
    WHERE jobs.embedding IS NOT NULL
    ORDER BY jobs.embedding <=> query_embedding
    LIMIT match_count;
$$;


-- Function: Find resumes matching a job
-- Returns top matching resumes based on embedding similarity
CREATE OR REPLACE FUNCTION match_resumes_to_job(
    query_embedding vector(768),
    match_count int DEFAULT 10
)
RETURNS TABLE (
    resume_id TEXT,
    name TEXT,
    email TEXT,
    professional_title TEXT,
    json_data JSONB,
    similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        resumes.resume_id,
        resumes.name,
        resumes.email,
        resumes.professional_title,
        resumes.json_data,
        1 - (resumes.embedding <=> query_embedding) as similarity
    FROM resumes
    WHERE resumes.embedding IS NOT NULL
    ORDER BY resumes.embedding <=> query_embedding
    LIMIT match_count;
$$;


-- Function: Search jobs by text query (generates embedding on the fly)
-- Note: This would require implementing embedding generation in Supabase
-- For now, embeddings should be generated client-side
COMMENT ON FUNCTION match_jobs_to_resume IS
'Find jobs matching a resume embedding. Pass the resume embedding vector and optionally specify match_count (default 10).';

COMMENT ON FUNCTION match_resumes_to_job IS
'Find resumes matching a job embedding. Pass the job embedding vector and optionally specify match_count (default 10).';


-- Optional: Add updated_at trigger for resumes
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_resumes_updated_at
    BEFORE UPDATE ON resumes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
