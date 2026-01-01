import modal
import os
from typing import List, Dict, Any
from datetime import datetime

# Create Modal app
app = modal.App("jobzilla-resume-embeddings")

# Reuse the same embedding image from job_embeddings.py
def download_nomic_model():
    """Download and cache the Nomic embedding model during image build."""
    from sentence_transformers import SentenceTransformer
    print("Downloading nomic-ai/nomic-embed-text-v1.5 model...")
    SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    print("Model download complete!")


embedding_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.35.0",
        "sentence-transformers==3.0.1",
        "numpy==1.24.3",
        "einops",
    )
    .run_function(download_nomic_model)
)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "supabase",
        "numpy==1.24.3",
    )
)


def json_resume_to_text(resume: Dict[str, Any]) -> str:
    """
    Convert JSON Resume format to searchable text for embedding.

    Focuses on content relevant for job matching:
    - Professional summary and title
    - Work experience with highlights
    - Skills and expertise
    - Education background
    - Projects and achievements

    Args:
        resume: Dictionary following JSON Resume schema (https://jsonresume.org/schema/)

    Returns:
        Formatted text string optimized for semantic search
    """
    parts = []

    # Basics: Professional identity and summary
    if basics := resume.get("basics", {}):
        name = basics.get("name", "")
        if name:
            parts.append(f"Name: {name}")

        label = basics.get("label", "")
        if label:
            parts.append(f"Professional Title: {label}")

        summary = basics.get("summary", "")
        if summary:
            parts.append(f"\nProfessional Summary:\n{summary}")

        # Location (useful for geographic matching)
        if location := basics.get("location", {}):
            city = location.get("city", "")
            country = location.get("countryCode", "")
            if city or country:
                parts.append(f"\nLocation: {city}, {country}")

    # Work Experience: Most important for job matching
    if work := resume.get("work", []):
        parts.append("\n\nWork Experience:")
        for job in work:
            position = job.get("position", "")
            company = job.get("name", "")
            start_date = job.get("startDate", "")
            end_date = job.get("endDate", "Present")
            summary = job.get("summary", "")
            highlights = job.get("highlights", [])

            job_text = f"\n{position} at {company}"
            if start_date:
                job_text += f" ({start_date} - {end_date})"

            if summary:
                job_text += f"\n{summary}"

            if highlights:
                job_text += "\n" + "\n".join(f"• {h}" for h in highlights)

            parts.append(job_text)

    # Skills: Critical for matching technical requirements
    if skills := resume.get("skills", []):
        parts.append("\n\nSkills and Expertise:")
        for skill in skills:
            skill_name = skill.get("name", "")
            level = skill.get("level", "")
            keywords = skill.get("keywords", [])

            skill_text = f"\n{skill_name}"
            if level:
                skill_text += f" ({level})"
            if keywords:
                skill_text += f": {', '.join(keywords)}"

            parts.append(skill_text)

    # Education: Important for role requirements
    if education := resume.get("education", []):
        parts.append("\n\nEducation:")
        for edu in education:
            institution = edu.get("institution", "")
            area = edu.get("area", "")
            study_type = edu.get("studyType", "")
            start_date = edu.get("startDate", "")
            end_date = edu.get("endDate", "")
            gpa = edu.get("gpa", "")
            courses = edu.get("courses", [])

            edu_text = f"\n{study_type} in {area} from {institution}"
            if start_date and end_date:
                edu_text += f" ({start_date} - {end_date})"
            if gpa:
                edu_text += f" - GPA: {gpa}"
            if courses:
                edu_text += f"\nRelevant courses: {', '.join(courses)}"

            parts.append(edu_text)

    # Projects: Shows practical application of skills
    if projects := resume.get("projects", []):
        parts.append("\n\nProjects:")
        for proj in projects:
            name = proj.get("name", "")
            description = proj.get("description", "")
            highlights = proj.get("highlights", [])
            keywords = proj.get("keywords", [])
            url = proj.get("url", "")

            proj_text = f"\n{name}"
            if description:
                proj_text += f"\n{description}"
            if highlights:
                proj_text += "\n" + "\n".join(f"• {h}" for h in highlights)
            if keywords:
                proj_text += f"\nTechnologies: {', '.join(keywords)}"

            parts.append(proj_text)

    # Awards and Recognition
    if awards := resume.get("awards", []):
        parts.append("\n\nAwards and Recognition:")
        for award in awards:
            title = award.get("title", "")
            date = award.get("date", "")
            awarder = award.get("awarder", "")
            summary = award.get("summary", "")

            award_text = f"\n{title}"
            if awarder:
                award_text += f" from {awarder}"
            if date:
                award_text += f" ({date})"
            if summary:
                award_text += f"\n{summary}"

            parts.append(award_text)

    # Publications: Relevant for research/academic roles
    if publications := resume.get("publications", []):
        parts.append("\n\nPublications:")
        for pub in publications:
            name = pub.get("name", "")
            publisher = pub.get("publisher", "")
            release_date = pub.get("releaseDate", "")
            summary = pub.get("summary", "")

            pub_text = f"\n{name}"
            if publisher:
                pub_text += f" - {publisher}"
            if release_date:
                pub_text += f" ({release_date})"
            if summary:
                pub_text += f"\n{summary}"

            parts.append(pub_text)

    # Languages: Important for international roles
    if languages := resume.get("languages", []):
        parts.append("\n\nLanguages:")
        lang_list = []
        for lang in languages:
            language = lang.get("language", "")
            fluency = lang.get("fluency", "")
            if language:
                lang_text = language
                if fluency:
                    lang_text += f" ({fluency})"
                lang_list.append(lang_text)
        if lang_list:
            parts.append(", ".join(lang_list))

    # Volunteer: Shows additional skills and values
    if volunteer := resume.get("volunteer", []):
        parts.append("\n\nVolunteer Experience:")
        for vol in volunteer:
            position = vol.get("position", "")
            organization = vol.get("organization", "")
            summary = vol.get("summary", "")
            highlights = vol.get("highlights", [])

            vol_text = f"\n{position} at {organization}"
            if summary:
                vol_text += f"\n{summary}"
            if highlights:
                vol_text += "\n" + "\n".join(f"• {h}" for h in highlights)

            parts.append(vol_text)

    return "\n".join(parts)


@app.function(
    image=embedding_image,
    gpu="T4",
    timeout=600,
)
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for resume texts using Nomic's embedding model.

    Same model as job embeddings to ensure resumes and jobs live in the same
    embedding space for effective matching.

    Args:
        texts: List of resume text strings to embed

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

    print(f"Generating embeddings for {len(texts)} resumes using Nomic v1.5")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
    )

    return embeddings.tolist()


@app.function(
    image=base_image,
    secrets=[modal.Secret.from_name("supabase-secret")],
    timeout=300,
)
def store_resumes_in_supabase(resumes: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Store JSON Resume documents and their embeddings in Supabase.

    Args:
        resumes: List of JSON Resume dictionaries
        embeddings: List of embedding vectors corresponding to each resume
    """
    from supabase import create_client, Client

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]

    supabase: Client = create_client(supabase_url, supabase_key)

    print(f"Storing {len(resumes)} resumes in Supabase")

    records = []
    for resume, embedding in zip(resumes, embeddings):
        # Extract key metadata for quick access
        basics = resume.get("basics", {})
        name = basics.get("name", "")
        email = basics.get("email", "")
        label = basics.get("label", "")

        # Generate a simple ID from email or timestamp
        resume_id = email if email else f"resume_{datetime.utcnow().timestamp()}"

        record = {
            "resume_id": resume_id,
            "name": name,
            "email": email,
            "professional_title": label,
            "json_data": resume,  # Store full JSON Resume
            "embedding": embedding,
            "created_at": datetime.utcnow().isoformat(),
        }
        records.append(record)

    # Deduplicate by resume_id (email or generated ID)
    unique_records = {}
    for record in records:
        unique_records[record["resume_id"]] = record
    deduplicated_records = list(unique_records.values())

    print(f"Deduplicated {len(records)} resumes to {len(deduplicated_records)} unique resumes")

    try:
        supabase.table("resumes").upsert(
            deduplicated_records,
            on_conflict="resume_id"
        ).execute()
        print(f"Successfully stored {len(deduplicated_records)} unique resumes")
    except Exception as e:
        print(f"Error storing resumes: {e}")
        raise


@app.function(
    image=base_image,
    secrets=[modal.Secret.from_name("supabase-secret")],
    timeout=300,
)
def search_matching_jobs(resume_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Find jobs that match a resume using vector similarity search.

    Args:
        resume_embedding: The embedding vector of a resume
        top_k: Number of top matching jobs to return

    Returns:
        List of matching jobs with similarity scores
    """
    from supabase import create_client, Client

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ["SUPABASE_KEY"]

    supabase: Client = create_client(supabase_url, supabase_key)

    # Use Supabase's vector similarity search (requires pgvector extension)
    # This will find jobs with embeddings closest to the resume embedding
    result = supabase.rpc(
        'match_jobs_to_resume',
        {
            'query_embedding': resume_embedding,
            'match_count': top_k
        }
    ).execute()

    return result.data


@app.local_entrypoint()
def main(resume_file: str = "sample_resume.json"):
    """
    Main pipeline: Load JSON Resume, generate embedding, store in Supabase.

    Usage:
        modal run resume_embeddings.py
        modal run resume_embeddings.py --resume-file path/to/resume.json
    """
    import json

    print(f"Starting resume embeddings pipeline for: {resume_file}")

    # Step 1: Load JSON Resume file
    print(f"\n[1/3] Loading resume from {resume_file}...")
    try:
        with open(resume_file, 'r') as f:
            resume = json.load(f)
        print(f"Loaded resume for: {resume.get('basics', {}).get('name', 'Unknown')}")
    except FileNotFoundError:
        print(f"Error: Resume file '{resume_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{resume_file}'")
        return

    # Step 2: Convert to text and generate embedding
    print("\n[2/3] Generating embedding...")
    resume_text = json_resume_to_text(resume)
    print(f"Resume text length: {len(resume_text)} characters")

    embeddings = generate_embeddings.remote([resume_text])

    # Step 3: Store in Supabase
    print("\n[3/3] Storing resume in Supabase...")
    store_resumes_in_supabase.remote([resume], embeddings)

    print("\n✓ Pipeline complete!")
    print(f"Resume stored with embedding dimension: {len(embeddings[0])}")
