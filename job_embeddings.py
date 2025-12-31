import modal
import os
from typing import List, Dict, Any
from datetime import datetime

# Create Modal app
app = modal.App("jobzilla-embeddings")

# Download Nomic model during image build to avoid cold start delays
def download_nomic_model():
    """Download and cache the Nomic embedding model during image build."""
    from sentence_transformers import SentenceTransformer
    print("Downloading nomic-ai/nomic-embed-text-v1.5 model...")
    SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    print("Model download complete!")


# Define image with embedding model dependencies
# Bake model weights into the Docker image at build time
embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.35.0",
        "sentence-transformers==3.0.1",  # Upgraded for trust_remote_code support
        "numpy==1.24.3",
        "einops",  # Required by nomic-ai/nomic-embed-text-v1.5
    )
    .run_function(download_nomic_model)  # Cache model in image
)

# Image for general tasks (API calls, DB operations)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "supabase",  # Use latest version for publishable key support
        "numpy==1.24.3",
    )
)


@app.function(
    image=base_image,
    secrets=[modal.Secret.from_name("rapidapi-secret")],
    timeout=300,
)
def fetch_jobs_from_jsearch(query: str = "software engineer", num_pages: int = 1) -> List[Dict[str, Any]]:
    """
    Fetch job listings from JSearch API (RapidAPI).

    Args:
        query: Search query for jobs
        num_pages: Number of pages to fetch (each page has ~10 jobs)

    Returns:
        List of job dictionaries with title, description, and metadata
    """
    import httpx

    rapidapi_key = os.environ["RAPIDAPI_KEY"]

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    all_jobs = []

    for page in range(1, num_pages + 1):
        params = {
            "query": query,
            "page": str(page),
            "num_pages": "1",
        }

        print(f"Fetching page {page} for query: {query}")

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            jobs = data.get("data", [])
            print(f"Found {len(jobs)} jobs on page {page}")
            all_jobs.extend(jobs)

    print(f"Total jobs fetched: {len(all_jobs)}")
    return all_jobs


@app.function(
    image=embedding_image,
    gpu="T4",  # Using T4 GPU for cost-effective embedding generation
    timeout=600,
)
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using Nomic's embedding model.

    Model: nomic-ai/nomic-embed-text-v1.5
    - 8192 token context window (handles 2-3 page documents)
    - 137M parameters
    - Optimized for semantic search and retrieval

    Args:
        texts: List of text strings to embed (job descriptions, resumes, etc.)

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load the Nomic model (already cached in image from build)
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

    print(f"Generating embeddings for {len(texts)} texts using Nomic v1.5")

    # Generate embeddings with batching for GPU efficiency
    # sentence-transformers handles batching automatically based on available VRAM
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,  # Adjust based on T4 VRAM (16GB) and doc size
    )

    # Convert to list of lists for JSON serialization
    return embeddings.tolist()


@app.function(
    image=base_image,
    secrets=[modal.Secret.from_name("supabase-secret")],
    timeout=300,
)
def store_jobs_in_supabase(jobs: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Store job listings and their embeddings in Supabase.

    Args:
        jobs: List of job dictionaries from JSearch
        embeddings: List of embedding vectors corresponding to each job
    """
    from supabase import create_client, Client

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]

    supabase: Client = create_client(supabase_url, supabase_key)

    print(f"Storing {len(jobs)} jobs in Supabase")

    # Prepare data for insertion
    records = []
    for job, embedding in zip(jobs, embeddings):
        # Combine title and description for the embedded text
        title = job.get("job_title", "")
        description = job.get("job_description", "")

        record = {
            "job_id": job.get("job_id"),
            "title": title,
            "description": description,
            "company": job.get("employer_name"),
            "location": job.get("job_city"),
            "country": job.get("job_country"),
            "employment_type": job.get("job_employment_type"),
            "posted_at": job.get("job_posted_at_datetime_utc"),
            "apply_link": job.get("job_apply_link"),
            "embedding": embedding,
            "fetched_at": datetime.utcnow().isoformat(),
        }
        records.append(record)

    # Insert into Supabase (assuming table name is 'jobs')
    # Using upsert to handle duplicates by job_id
    try:
        supabase.table("jobs").upsert(
            records,
            on_conflict="job_id"
        ).execute()
        print(f"Successfully stored {len(records)} jobs")
    except Exception as e:
        print(f"Error storing jobs: {e}")
        raise


@app.local_entrypoint()
def main(query: str = "software engineer", num_pages: int = 1):
    """
    Main pipeline: Fetch jobs, generate embeddings, store in Supabase.

    Usage:
        modal run job_embeddings.py
        modal run job_embeddings.py --query "python developer" --num-pages 3
    """
    print(f"Starting job embeddings pipeline for query: '{query}'")

    # Step 1: Fetch jobs from JSearch
    print("\n[1/3] Fetching jobs from JSearch...")
    jobs = fetch_jobs_from_jsearch.remote(query=query, num_pages=num_pages)

    if not jobs:
        print("No jobs found. Exiting.")
        return

    # Step 2: Prepare texts for embedding (title + description)
    print("\n[2/3] Generating embeddings...")
    texts = []
    for job in jobs:
        title = job.get("job_title", "")
        description = job.get("job_description", "")
        # Combine title and description
        combined_text = f"{title}\n\n{description}"
        texts.append(combined_text)

    # Generate embeddings in batches
    embeddings = generate_embeddings.remote(texts)

    # Step 3: Store in Supabase
    print("\n[3/3] Storing jobs in Supabase...")
    store_jobs_in_supabase.remote(jobs, embeddings)

    print("\n✓ Pipeline complete!")
    print(f"Processed {len(jobs)} jobs")
