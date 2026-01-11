# KDSH Track A – Narrative Consistency Classification

## Overview
This project addresses Track A of the Kharagpur Data Science Hackathon 2026.
The task is to determine whether a hypothetical character backstory is
consistent with a long-form narrative (novel).

## Approach
- Full novels are ingested and chunked to handle long context.
- Sentence embeddings are generated using SentenceTransformers.
- A FAISS vector index enables retrieval over novel chunks.
- Train data is used for pattern grounding.
- Test backstories are classified for consistency.

## Folder Structure
- books/        : Full novels (.txt)
- data/         : train.csv, test.csv
- code/         : main pipeline and utilities
- results.csv   : Final predictions

## How to Run
```bash
pip install -r code/requirements.txt
cd code
python main.py
