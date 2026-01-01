import modal
import os
from typing import List, Dict, Any
from datetime import datetime

# Create Modal app
app = modal.App("jobzilla-embeddings-adzuna")

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
    secrets=[modal.Secret.from_name("adzuna-secret")],
    timeout=300,
)
def fetch_jobs_from_adzuna(
    what: str = "ai engineer",
    where: str = "london",
    country: str = "gb",
    num_pages: int = 10,
    sort_by: str = "date",
) -> List[Dict[str, Any]]:
    """
    Fetch job listings from Adzuna API.

    Adzuna API docs: https://developer.adzuna.com/docs/search

    Args:
        what: Job title or keywords to search for
        where: Location (city, region, etc.)
        country: Country code (gb, us, au, etc.)
        num_pages: Number of pages to fetch (each page has up to 50 jobs)
        sort_by: Sort order - 'date' (freshest), 'salary' (highest paid), or omit for relevance

    Returns:
        List of job dictionaries with title, description, and metadata
    """
    import httpx
    import time

    app_id = os.environ["ADZUNA_APP_ID"]
    app_key = os.environ["ADZUNA_APP_KEY"]

    base_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search"

    all_jobs = []
    consecutive_empty_pages = 0

    for page in range(1, num_pages + 1):
        url = f"{base_url}/{page}"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": what,
            "where": where,
            "results_per_page": 50,  # Max results per page
            "sort_by": sort_by,
            "content-type": "application/json",
        }

        print(f"Fetching page {page} for '{what}' in '{where}'")

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            jobs = data.get("results", [])
            print(f"Found {len(jobs)} jobs on page {page}")

            if not jobs:
                consecutive_empty_pages += 1
                # Stop if we hit 2 consecutive empty pages
                if consecutive_empty_pages >= 2:
                    print(f"Stopping early: {consecutive_empty_pages} consecutive empty pages")
                    break
            else:
                consecutive_empty_pages = 0
                all_jobs.extend(jobs)

        # Add delay between requests to be respectful to API
        if page < num_pages:
            time.sleep(0.5)

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
        jobs: List of job dictionaries from Adzuna
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
        # Extract Adzuna-specific fields
        title = job.get("title", "")
        description = job.get("description", "")

        # Adzuna uses different field names than JSearch
        record = {
            "job_id": job.get("id"),  # Adzuna uses 'id' instead of 'job_id'
            "title": title,
            "description": description,
            "company": job.get("company", {}).get("display_name") if isinstance(job.get("company"), dict) else None,
            "location": job.get("location", {}).get("display_name") if isinstance(job.get("location"), dict) else None,
            "country": job.get("location", {}).get("area", [None])[0] if isinstance(job.get("location"), dict) else None,
            "employment_type": job.get("contract_type"),
            "posted_at": job.get("created"),
            "apply_link": job.get("redirect_url"),
            "salary_min": job.get("salary_min"),
            "salary_max": job.get("salary_max"),
            "salary_is_predicted": job.get("salary_is_predicted"),
            "source": "Adzuna",
            "embedding": embedding,
            "fetched_at": datetime.utcnow().isoformat(),
        }
        records.append(record)

    # Deduplicate records by job_id (keep last occurrence)
    unique_records = {}
    for record in records:
        unique_records[record["job_id"]] = record
    deduplicated_records = list(unique_records.values())

    print(f"Deduplicated {len(records)} jobs to {len(deduplicated_records)} unique jobs")

    # Insert into Supabase (assuming table name is 'jobs')
    # Using upsert to handle duplicates by job_id
    try:
        supabase.table("jobs").upsert(
            deduplicated_records,
            on_conflict="job_id"
        ).execute()
        print(f"Successfully stored {len(deduplicated_records)} unique jobs")
    except Exception as e:
        print(f"Error storing jobs: {e}")
        raise


@app.local_entrypoint()
def main(
    what: str = "ai engineer",
    where: str = "london",
    country: str = "gb",
    num_pages: int = 10,
    sort_by: str = "date",
):
    """
    Main pipeline: Fetch jobs from Adzuna, generate embeddings, store in Supabase.

    Usage:
        modal run job_embeddings_adzuna.py
        modal run job_embeddings_adzuna.py --what "python developer" --where "manchester" --num-pages 5
        modal run job_embeddings_adzuna.py --what "ai engineer" --sort-by "salary"
    """
    print(f"Starting Adzuna job embeddings pipeline for '{what}' in '{where}' ({country})")
    print(f"Sorting by: {sort_by}")

    # Step 1: Fetch jobs from Adzuna
    print("\n[1/3] Fetching jobs from Adzuna...")
    jobs = fetch_jobs_from_adzuna.remote(
        what=what,
        where=where,
        country=country,
        num_pages=num_pages,
        sort_by=sort_by,
    )

    if not jobs:
        print("No jobs found. Exiting.")
        return

    # Step 2: Prepare texts for embedding (title + description)
    print("\n[2/3] Generating embeddings...")
    texts = []
    for job in jobs:
        title = job.get("title", "")
        description = job.get("description", "")
        # Combine title and description
        combined_text = f"{title}\n\n{description}"
        texts.append(combined_text)

    # Generate embeddings in batches
    embeddings = generate_embeddings.remote(texts)

    # Step 3: Store in Supabase
    print("\n[3/3] Storing jobs in Supabase...")
    store_jobs_in_supabase.remote(jobs, embeddings)

    print("\n✓ Pipeline complete!")
    print(f"Processed {len(jobs)} jobs from Adzuna")
