from rag import index_pdf

if __name__ == "__main__":
    embeddings, chunks = index_pdf(
        "Introduction_to_Networks.pdf",
        doc_id="introduction_to_networks"
    )

    print("Index created successfully!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")