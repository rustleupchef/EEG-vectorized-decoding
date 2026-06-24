import lancedb
from datasets import load_dataset

def main():
    # 1. Create/Connect to a local directory to store your vectors
    db = lancedb.connect("./wiki_vectors_db")

    # 2. Load the 35M row dataset via streaming so it uses almost zero RAM
    print("Connecting to Hugging Face dataset...")
    hf_dataset = load_dataset(
        "maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2", 
        split="train", 
        streaming=True
    )

    # 3. Create a generator to map Hugging Face columns to LanceDB format
    # Note: LanceDB expects the vector column to be explicitly named 'vector'
    def get_batches(dataset, batch_size=25000):
        batch = []
        for row in dataset:
            batch.append({
                "vector": row["emb"],  # The precomputed 384-d all-MiniLM-L6-v2 array
                "text": row["text"],   # The original English text snippet
                "title": row["title"]  # Article title metadata
            })
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    batches = get_batches(hf_dataset)

    # 4. Initialize your local database table with the first chunk
    print("Initializing database table...")
    table = db.create_table("wikipedia", data=next(batches), mode="overwrite")

    # 5. Loop through and pull down the remaining data
    print("Streaming vectors to disk. (You can stop this early if you just want a sample)...")
    for i, chunk in enumerate(batches):
        table.add(chunk)
        if i % 10 == 0:
            print(f"Indexed {(i+1) * 25000} rows...")
        if (i + 1) * 25000 >= 2_000_000:
            break 

    print("Ingestion complete! Your universal concept-to-text database is ready.")

if __name__ == "__main__":
    main()