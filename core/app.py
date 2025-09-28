# app.py

import os
import streamlit as st
import tempfile
import zipfile
import io

from build_wiki import build_sharded_wiki_from_github, get_unique_cache_dir
from build_embeddings import build_units_for_repo, summarize_units_with_qgenie, units_to_dataframe, embed_texts, build_faiss_index, save_index, ensure_dir
from generate_wiki_pages import generate_wiki_pages
from hybrid_code_chatbot import HybridCodeChatbot

# ========== UI: Sidebar Inputs ==========
st.set_page_config(page_title="Codebase Wiki & Chatbot", layout="wide")
st.title("📚 Codebase Documentation & Q&A Chatbot")

with st.sidebar:
    st.header("Repository & Generation Settings")
    github_url = st.text_input("GitHub URL", value="https://github.com/pallets/flask")
    github_token = st.text_input("GitHub Token (optional)", type="password")
    language = st.selectbox("Documentation Language", ["English", "Chinese", "Spanish", "French"], index=0)
    model_name = st.selectbox("QGenie Model", ["Pro", "qwen2.5-14b-1m"], index=0)
    regenerate = st.checkbox("Regenerate Documentation (ignore cache)", value=False)
    generate_btn = st.button("Generate Documentation", type="primary", use_container_width=True)

# ========== Pipeline Execution ==========
def run_pipeline(github_url, github_token, language, model_name, regenerate):
    repo_dir = get_unique_cache_dir(github_url)
    os.makedirs(repo_dir, exist_ok=True)
    cache_exists = os.path.exists(os.path.join(repo_dir, "wiki_pages"))

    # Use cache if available and not regenerating
    if cache_exists and not regenerate:
        st.success("Loaded documentation from cache.")
    else:
        with st.spinner("Building sharded wiki..."):
            build_sharded_wiki_from_github(
                gh_url=github_url,
                token=github_token,
                output_root=repo_dir,
                lang_text=language,
                qgenie_model=model_name,
            )
        with st.spinner("Building code embeddings..."):
            gid, meta, units = build_units_for_repo(github_url, token=github_token, output_root=repo_dir)
            embeddings_dir = os.path.join(repo_dir, "embeddings")
            ensure_dir(embeddings_dir)
            summarize_units_with_qgenie(units)
            df = units_to_dataframe(units)
            df_path = os.path.join(embeddings_dir, "units.parquet")
            df.to_parquet(df_path, index=False)
            file_mask = (df["level"] == "file")
            sym_mask = (df["level"] == "symbol")
            file_df = df[file_mask].copy()
            sym_df = df[sym_mask].copy()
            file_summary_texts = file_df["summary"].fillna("").tolist()
            sym_summary_texts = sym_df["summary"].fillna("").tolist()
            file_summary_vecs = embed_texts(file_summary_texts)
            sym_summary_vecs = embed_texts(sym_summary_texts)
            file_df["summary_embedding"] = file_summary_vecs.tolist()
            sym_df["summary_embedding"] = sym_summary_vecs.tolist()
            file_summary_index = build_faiss_index(file_summary_vecs)
            sym_summary_index = build_faiss_index(sym_summary_vecs)
            file_ids = file_df["uid"].tolist()
            symbol_ids = sym_df["uid"].tolist()
            save_index(file_summary_index, file_ids, embeddings_dir, "file_summary")
            save_index(sym_summary_index, symbol_ids, embeddings_dir, "symbol_summary")
        with st.spinner("Generating documentation pages..."):
            generate_wiki_pages(
                output_root=repo_dir,
                lang_text=language,
            )
        st.success("Documentation generated and cached.")

    # Load documentation and chatbot
    wiki_pages_dir = os.path.join(repo_dir, "wiki_pages")
    page_files = sorted([f for f in os.listdir(wiki_pages_dir) if f.endswith(".md")])
    page_contents = {}
    for f in page_files:
        with open(os.path.join(wiki_pages_dir, f), "r", encoding="utf-8") as fp:
            page_contents[f] = fp.read()
    chatbot = HybridCodeChatbot(os.path.join(repo_dir, "embeddings"))
    return page_files, page_contents, chatbot, repo_dir

# ========== Session State ==========
if "page_files" not in st.session_state:
    st.session_state.page_files = []
if "page_contents" not in st.session_state:
    st.session_state.page_contents = {}
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "repo_dir" not in st.session_state:
    st.session_state.repo_dir = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========== Run Pipeline ==========
if generate_btn:
    st.session_state.page_files, st.session_state.page_contents, st.session_state.chatbot, st.session_state.repo_dir = run_pipeline(
        github_url, github_token, language, model_name, regenerate
    )
    st.session_state.chat_history = []

# ========== Documentation UI ==========
if st.session_state.page_files:
    col1, col2 = st.columns([2, 2], gap="large")
    with col1:
        st.header("📑 Documentation Pages")
        selected_page = st.radio(
            "Select a documentation page:",
            st.session_state.page_files,
            format_func=lambda x: x.replace(".md", "").replace("_", " ").title(),
        )
        st.markdown("---")
        st.markdown(st.session_state.page_contents[selected_page], unsafe_allow_html=True)
        # Download all docs as ZIP
        if st.button("Download All Documentation", use_container_width=True):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, content in st.session_state.page_contents.items():
                    zf.writestr(fname, content)
            st.download_button(
                label="Download ZIP",
                data=buf.getvalue(),
                file_name="documentation.zip",
                mime="application/zip",
                use_container_width=True,
            )
    with col2:
        st.header("💬 Codebase Q&A Chatbot")
        user_question = st.text_input("Ask your question here:", key="chat_input")
        if st.button("Ask", use_container_width=True):
            if user_question.strip():
                # Hybrid retrieval
                retrieval = st.session_state.chatbot.retrieve(user_question, topk_file=5, topk_symbol=5)
                context = retrieval["context"]
                # Call QGenie for answer
                from qgenie import QGenieClient, ChatMessage
                client = QGenieClient()
                prompt = f"""You are a codebase assistant. Use the following context to answer the user's question.

Context:
{context}

Question:
{user_question}

Return a concise, factual answer. If relevant, cite file paths or symbol names from the context."""
                response = client.chat(messages=[ChatMessage(role="user", content=prompt)], model=model_name)
                answer = getattr(response, "first_content", None) or str(response)
                # Save to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer,
                    "references": retrieval["files"] + retrieval["symbols"],
                })
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for idx, entry in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{idx+1}:** {entry['question']}")
                st.markdown(f"**A{idx+1}:** {entry['answer']}")
                if entry["references"]:
                    st.markdown("**References:**")
                    for ref in entry["references"]:
                        st.markdown(f"- `{ref.get('file_path', '')}` {ref.get('symbol_type', '')} {ref.get('symbol_name', '')}")

else:
    st.info("Enter a GitHub URL and click **Generate Documentation** to begin.")
