def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Splits long text into overlapping chunks.
    This helps handle very large novels.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
