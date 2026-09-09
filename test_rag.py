from src.slam_llm.rag import RAGRetriever

rag = RAGRetriever()

print("Loading documents...")
chunks = rag.load_documents("rag_data")
print("Loaded chunks:", len(chunks))

print("Building index...")
rag.build_index()

query = "How does RAG help SmartSLAM?"
print("\nQuery:", query)

results = rag.retrieve(query, top_k=3)

print("\n=== RETRIEVED EVIDENCE ===")
print(results)

print("\n=== CONFIDENCE ===")
print(rag.confidence(results))

print("\n=== EXPLANATION ===")
print(rag.explain(results))

print("\n=== AUGMENTED PROMPT ===")
print(rag.build_augmented_prompt(query, results))
