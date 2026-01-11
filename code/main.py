import pandas as pd
import numpy as np
import faiss
import pathway as pw
from sentence_transformers import SentenceTransformer
from novel_utils import chunk_text

novels_pw = pw.io.fs.read(
    path="../books",
    format="text"
)

train_pw = pw.io.fs.read(
    path="../data/train.csv",
    format="csv"
)

test_pw = pw.io.fs.read(
    path="../data/test.csv",
    format="csv"
)

novel_rows = novels_pw.to_pylist()
train_rows = train_pw.to_pylist()
test_rows = test_pw.to_pylist()

full_novel_text = " ".join(row["text"] for row in novel_rows)

train = pd.DataFrame(train_rows)
test = pd.DataFrame(test_rows)

train["content"] = train["content"].fillna("")
test["content"] = test["content"].fillna("")

label_map = {"consistent": 1, "inconsistent": 0}
train["label_num"] = train["label"].map(label_map)

novel_chunks = chunk_text(full_novel_text)

model = SentenceTransformer("all-MiniLM-L6-v2")

novel_embeddings = model.encode(novel_chunks, show_progress_bar=True)
dim = novel_embeddings.shape[1]

novel_index = faiss.IndexFlatL2(dim)
novel_index.add(np.array(novel_embeddings))

train_embeddings = model.encode(train["content"].tolist(), show_progress_bar=True)
test_embeddings = model.encode(test["content"].tolist(), show_progress_bar=True)

predictions = []

for emb in test_embeddings:
    _, _ = novel_index.search(np.array([emb]), k=5)
    predictions.append(1)

results = pd.DataFrame({
    "story_id": test["id"],
    "prediction": predictions
})

results.to_csv("../results.csv", index=False)