# hybrid_code_chatbot.py

import os
import json
import numpy as np
import pandas as pd

class HybridCodeChatbot:
    """
    Chatbot for querying hybrid code embeddings (file-level + symbol-level) built by build_embeddings.py.
    Loads FAISS indices and metadata, embeds user queries, retrieves top relevant units.
    """
    def __init__(self, embeddings_dir: str):
        """
        embeddings_dir: Path to .cache/embeddings/<graph_id>/
        """
        import faiss

        # Load metadata and dataframe
        units_path = os.path.join(embeddings_dir, "units.parquet")
        if not os.path.isfile(units_path):
            raise FileNotFoundError(f"units.parquet not found in {embeddings_dir}")
        self.df = pd.read_parquet(units_path).set_index("uid")

        # Load file-level index and ids
        self.file_index = faiss.read_index(os.path.join(embeddings_dir, "file.index"))
        with open(os.path.join(embeddings_dir, "file_ids.json"), "r", encoding="utf-8") as f:
            self.file_ids = json.load(f)

        # Load symbol-level index and ids
        self.symbol_index = faiss.read_index(os.path.join(embeddings_dir, "symbol.index"))
        with open(os.path.join(embeddings_dir, "symbol_ids.json"), "r", encoding="utf-8") as f:
            self.symbol_ids = json.load(f)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed the query using QGenie.
        """
        from qgenie import QGenieClient
        client = QGenieClient()
        embedding_response = client.embeddings([query])
        embedding = embedding_response.data[0].embedding
        return np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    def retrieve(self, query: str, topk_file: int = 5, topk_symbol: int = 5) -> dict:
        """
        Retrieve top file-level and symbol-level units for the query.
        Returns: dict with 'files', 'symbols', and 'context' (markdown).
        """
        qvec = self.embed_query(query)
        Df, If = self.file_index.search(qvec, topk_file)
        Ds, Is = self.symbol_index.search(qvec, topk_symbol)

        file_hits = []
        for score, idx in zip(Df[0].tolist(), If[0].tolist()):
            uid = self.file_ids[idx]
            row = self.df.loc[uid]
            file_hits.append({
                "uid": uid,
                "file_path": row["file_path"],
                "summary": row.get("summary", ""),
                "code": row.get("code", ""),
                "score": float(score),
            })

        symbol_hits = []
        for score, idx in zip(Ds[0].tolist(), Is[0].tolist()):
            uid = self.symbol_ids[idx]
            row = self.df.loc[uid]
            symbol_hits.append({
                "uid": uid,
                "file_path": row["file_path"],
                "symbol_type": row.get("symbol_type", ""),
                "symbol_name": row.get("symbol_name", ""),
                "signature": row.get("signature", ""),
                "summary": row.get("summary", ""),
                "code": row.get("code", ""),
                "score": float(score),
            })

        # Compose context markdown for LLM answering
        context_md = []
        context_md.append("## Top Files\n")
        for hit in file_hits:
            context_md.append(f"**{hit['file_path']}**\n")
            if hit["summary"]:
                context_md.append(f"- Summary: {hit['summary']}\n")
            context_md.append(f"```python\n{hit['code'][:800]}\n```\n")

        context_md.append("\n## Top Symbols\n")
        for hit in symbol_hits:
            context_md.append(f"**{hit['symbol_type']} {hit['symbol_name']} ({hit['file_path']})**\n")
            if hit["summary"]:
                context_md.append(f"- Summary: {hit['summary']}\n")
            context_md.append(f"```python\n{hit['code'][:800]}\n```\n")

        return {
            "files": file_hits,
            "symbols": symbol_hits,
            "context": "\n".join(context_md)
        }

# Example usage:
if __name__ == "__main__":
    # Path to your embeddings directory
    embeddings_dir = ".cache/embeddings/<graph_id>"
    chatbot = HybridCodeChatbot(embeddings_dir)
    query = "How is request routing implemented?"
    result = chatbot.retrieve(query, topk_file=5, topk_symbol=5)
    print(result["context"])