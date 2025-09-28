import argparse
import os

# Import builder functions and cache dir utility
from build_wiki import build_sharded_wiki_from_github, get_unique_cache_dir
from build_embeddings import build_units_for_repo, summarize_units_with_qgenie, units_to_dataframe, embed_texts, build_faiss_index, save_index, ensure_dir
from generate_wiki_pages import generate_wiki_pages

def process_github_repo(github_url, token=None, MAX_FILES=None):
    # Compute unique cache directory for this repo
    repo_dir = get_unique_cache_dir(github_url)
    os.makedirs(repo_dir, exist_ok=True)

    print(f"Processing GitHub repo: {github_url}")
    print(f"Artifacts will be saved under: {repo_dir}")

    # Build wiki shards and knowledge graph
    print("Building knowledge graph and wiki shards...")
    wiki_result = build_sharded_wiki_from_github(
        gh_url=github_url,
        token=token,
        output_root=repo_dir  # <-- ensure this is supported in your builder
    )
    print("Wiki/Knowledge Graph build complete.")

    # Build code embeddings
    print("Building code embeddings...")
    gid, meta, units = build_units_for_repo(github_url, token=token, MAX_FILES=MAX_FILES)
    embeddings_dir = os.path.join(repo_dir, "embeddings")
    ensure_dir(embeddings_dir)

    # Summarize units (always)
    summarize_units_with_qgenie(units)

    # Save units as DataFrame
    import pandas as pd
    df = units_to_dataframe(units)
    df_path = os.path.join(embeddings_dir, "units.parquet")
    df.to_parquet(df_path, index=False)

    # Prepare embedding texts
    file_mask = (df["level"] == "file")
    sym_mask = (df["level"] == "symbol")
    file_df = df[file_mask].copy()
    sym_df = df[sym_mask].copy()
    file_summary_texts = file_df["summary"].fillna("").tolist()
    sym_summary_texts = sym_df["summary"].fillna("").tolist()

    # Embeddings for summaries
    file_summary_vecs = embed_texts(file_summary_texts)
    sym_summary_vecs = embed_texts(sym_summary_texts)

    # Save summary embeddings in dataframe
    file_df["summary_embedding"] = file_summary_vecs.tolist()
    sym_df["summary_embedding"] = sym_summary_vecs.tolist()

    # FAISS indices
    file_summary_index = build_faiss_index(file_summary_vecs)
    sym_summary_index = build_faiss_index(sym_summary_vecs)

    # Save indices + ids
    file_ids = file_df["uid"].tolist()
    symbol_ids = sym_df["uid"].tolist()
    save_index(file_summary_index, file_ids, embeddings_dir, "file_summary")
    save_index(sym_summary_index, symbol_ids, embeddings_dir, "symbol_summary")

    print("\nSummary of outputs:")
    print(f"Wiki/Knowledge Graph: {wiki_result}")
    print(f"Embeddings directory: {embeddings_dir}")

    # === Generate documentation pages ===
    print("Generating documentation pages...")
    wiki_xml_path = os.path.join(repo_dir, "wiki_xml", "wiki.xml")
    result = generate_wiki_pages(
        output_root=repo_dir,
        wiki_xml_path=wiki_xml_path,
        embeddings_dir=embeddings_dir,
        # Optionally, pass owner/repo/branch/subpath if you want clickable references
        # owner="your_owner", repo="your_repo", branch="main", subpath=""
    )
    print(f"Documentation pages generated in: {result['wiki_pages_dir']}")
    print(f"All artifacts saved under: {repo_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build wiki shards, code embeddings, and documentation for a GitHub repo.")
    parser.add_argument("--github-url", required=True, help="GitHub repository URL")
    parser.add_argument("--token", default=None, help="GitHub access token (optional)")
    parser.add_argument("--max-files", type=int, default=2, help="Maximum number of files to process for testing")
    args = parser.parse_args()

    process_github_repo(args.github_url, token=args.token, MAX_FILES=args.max_files)


