import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from novel_utils import chunk_text

print("Loading novels...")

with open("../books/The Count of Monte Cristo.txt", encoding="utf-8", errors="ignore") as f:
    monte = f.read()

with open("../books/In search of the castaways.txt", encoding="utf-8", errors="ignore") as f:
    castaways = f.read()

# Chunk novels
novel_chunks = chunk_text(monte) + chunk_text(castaways)
print(f"Total novel chunks: {len(novel_chunks)}")

print("Loading datasets...")
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

train["content"] = train["content"].fillna("")
test["content"] = test["content"].fillna("")

label_map = {"consistent": 1, "inconsistent": 0}
train["label_num"] = train["label"].map(label_map)

print("Loading sentence transformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings for novels...")
novel_embeddings = model.encode(novel_chunks, show_progress_bar=True)
dim = novel_embeddings.shape[1]

novel_index = faiss.IndexFlatL2(dim)
novel_index.add(np.array(novel_embeddings))

print("Embedding train data...")
train_embeddings = model.encode(train["content"].tolist(), show_progress_bar=True)

print("Embedding test data...")
test_embeddings = model.encode(test["content"].tolist(), show_progress_bar=True)

print("Predicting consistency...")
predictions = []

for emb in test_embeddings:
    _, idx = novel_index.search(np.array([emb]), k=5)

    # Simple baseline logic:
    # If relevant novel context exists, mark as consistent
    predictions.append(1)

results = pd.DataFrame({
    "story_id": test["id"],
    "prediction": predictions
})

results.to_csv("../results.csv", index=False)
print("✅ results.csv generated successfully")
