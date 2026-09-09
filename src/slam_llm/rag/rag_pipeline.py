from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class DocumentChunk:
    source: str
    chunk_id: int
    text: str


class RAGRetriever:
    """
    Lightweight Retrieval-Augmented Generation retriever.

    Pipeline:
        Documents
            -> Chunking
            -> Embeddings
            -> In-memory vector index
            -> Cosine similarity retrieval
            -> Context construction
            -> Augmented prompt
    """

    def __init__(
        self,
        chunk_size: int = 120,
        chunk_overlap: int = 20,
        device: str = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.embedding_model_name = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Loading embedding model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name
        )

        self.model = AutoModel.from_pretrained(
            self.embedding_model_name
        ).to(self.device)

        self.model.eval()

        self.chunks = []
        self.embeddings = None

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean-pool token embeddings while ignoring padding tokens.
        """
        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return torch.sum(
            token_embeddings * input_mask_expanded,
            dim=1,
        ) / torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9,
        )

    def encode(self, texts):
        """
        Convert text into normalized vector embeddings.
        """
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded_input = {
            key: value.to(self.device)
            for key, value in encoded_input.items()
        }

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self.mean_pooling(
            model_output,
            encoded_input["attention_mask"],
        )

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        return embeddings.cpu()

    def chunk_text(self, text):
        """
        Split a document into overlapping word-based chunks.
        """
        words = text.split()

        if not words:
            return []

        chunks = []

        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(
                start + self.chunk_size,
                len(words),
            )

            chunk = " ".join(words[start:end])

            chunks.append(
                DocumentChunk(
                    source="",
                    chunk_id=chunk_id,
                    text=chunk,
                )
            )

            chunk_id += 1

            if end >= len(words):
                break

            start = end - self.chunk_overlap

        return chunks

    def load_documents(self, directory):
        """
        Load .txt documents and split them into chunks.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Document directory not found: {directory}"
            )

        self.chunks = []

        for file_path in sorted(directory.glob("*.txt")):
            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                continue

            document_chunks = self.chunk_text(text)

            for chunk in document_chunks:
                chunk.source = file_path.name
                self.chunks.append(chunk)

        if not self.chunks:
            raise ValueError(
                f"No text documents found in {directory}"
            )

        return self.chunks

    def build_index(self):
        """
        Generate embeddings and build an in-memory vector index.
        """
        if not self.chunks:
            raise ValueError(
                "No document chunks available. "
                "Run load_documents() first."
            )

        texts = [
            chunk.text
            for chunk in self.chunks
        ]

        self.embeddings = self.encode(texts)

        return self.embeddings

    def retrieve(
        self,
        query,
        top_k: int = 3,
        min_score: float = 0.15,
    ):
        """
        Retrieve the most relevant document chunks using
        cosine similarity.
        """
        if self.embeddings is None:
            raise ValueError(
                "Index has not been built. "
                "Run build_index() first."
            )

        query_embedding = self.encode([query])[0]

        scores = torch.matmul(
            self.embeddings,
            query_embedding,
        )

        top_k = min(
            top_k,
            len(self.chunks),
        )

        values, indices = torch.topk(
            scores,
            k=top_k,
        )

        results = []

        for score, index in zip(values, indices):
            similarity = float(score.item())

            if similarity < min_score:
                continue

            chunk = self.chunks[int(index.item())]

            results.append(
                {
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "similarity": similarity,
                }
            )

        return results

    def build_context(self, results):
        """
        Convert retrieved evidence into context for the LLM.
        """
        if not results:
            return "No relevant evidence was retrieved."

        context_parts = []

        for result in results:
            context_parts.append(
                f"[Source: {result['source']} | "
                f"Similarity: {result['similarity']:.4f}]\n"
                f"{result['text']}"
            )

        return "\n\n".join(context_parts)

    def build_augmented_prompt(self, query, results):
        """
        Construct an evidence-grounded prompt.
        """
        context = self.build_context(results)

        return (
            "Answer the question using the retrieved evidence "
            "below.\n\n"
            "Retrieved Evidence:\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using the available evidence. "
            "If the evidence does not contain the answer, "
            "say that the available evidence is insufficient."
        )

    def confidence(self, results, top_k=3):
        """
        Calculate an engineering confidence score.

        Confidence combines:
        1. Average retrieval similarity.
        2. Amount of retrieved supporting evidence.
        """
        if not results:
            return 0.0

        similarities = [
            float(result["similarity"])
            for result in results
        ]

        average_similarity = sum(similarities) / len(
            similarities
        )

        retrieval_strength = max(
            0.0,
            min(1.0, average_similarity),
        )

        evidence_score = min(
            1.0,
            len(results) / max(1, top_k),
        )

        confidence_score = (
            0.7 * retrieval_strength
            + 0.3 * evidence_score
        )

        return round(
            max(0.0, min(1.0, confidence_score)),
            4,
        )

    def explain(self, results):
        """
        Provide observable XAI evidence.

        This does not expose hidden chain-of-thought.
        It reports the documents, passages and similarity
        scores used as external evidence.
        """
        if not results:
            return (
                "No relevant evidence was retrieved. "
                "The answer has no supporting document evidence."
            )

        explanation_lines = [
            f"Retrieved {len(results)} supporting "
            f"evidence chunk(s)."
        ]

        for result in results:
            explanation_lines.append(
                f"- Source: {result['source']}, "
                f"Chunk: {result['chunk_id']}, "
                f"Similarity: {result['similarity']:.4f}"
            )

        explanation_lines.append(
            "Support was determined from retrieved "
            "document evidence and similarity scores."
        )

        return "\n".join(explanation_lines)

    def get_xai_metadata(self, results):
        """
        Return structured explainability information.
        """
        return {
            "evidence_available": bool(results),
            "num_evidence_chunks": len(results),
            "sources": [
                result["source"]
                for result in results
            ],
            "similarity_scores": [
                round(float(result["similarity"]), 4)
                for result in results
            ],
            "explanation": self.explain(results),
            "confidence": self.confidence(results),
        }