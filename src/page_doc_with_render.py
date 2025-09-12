import os
import io
import re
import json
import gzip
import uuid
import zipfile
import hashlib
import traceback
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ===================== Load environment variables =====================

load_dotenv()
# ========================= FS Utilities =========================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ========================= XML Parsing =========================

def load_wiki_xml(path: str) -> Dict[str, Any]:
    """
    Parse minimal structure:
      sections: [{id, title, page_refs[], subsections[]}]
      pages: {page-id: {title, description, importance, relevant_files[], related_pages[], parent_section}}
    """
    import xml.etree.ElementTree as ET
    text = read_text(path).strip()
    start = text.find("<wiki_structure")
    end = text.rfind("</wiki_structure>")
    if start != -1 and end != -1:
        text = text[start:end + len("</wiki_structure>")]
    root = ET.fromstring(text)
    if root.tag != "wiki_structure":
        raise ValueError("Root is not <wiki_structure>")

    sections_el = root.find("sections")
    pages_el = root.find("pages")
    sections = []
    pages_map = {}

    if sections_el is not None:
        for s in sections_el.findall("section"):
            sid = s.get("id") or ""
            title = (s.findtext("title") or "").strip()
            refs = []
            pages_container = s.find("pages")
            if pages_container is not None:
                for pr in pages_container.findall("page_ref"):
                    tid = (pr.text or "").strip()
                    if tid:
                        refs.append(tid)
            subs = []
            subs_el = s.find("subsections")
            if subs_el is not None:
                for sr in subs_el.findall("section_ref"):
                    t = (sr.text or "").strip()
                    if t:
                        subs.append(t)
            sections.append({"id": sid, "title": title, "page_refs": refs, "subsections": subs})

    if pages_el is not None:
        for p in pages_el.findall("page"):
            pid = p.get("id") or ""
            title = (p.findtext("title") or "").strip()
            desc = (p.findtext("description") or "").strip()
            imp = (p.findtext("importance") or "medium").strip().lower()
            rf = []
            rf_el = p.find("relevant_files")
            if rf_el is not None:
                for fp in rf_el.findall("file_path"):
                    t = (fp.text or "").strip()
                    if t:
                        rf.append(t)
            rel = []
            rel_el = p.find("related_pages")
            if rel_el is not None:
                for rp in rel_el.findall("related"):
                    t = (rp.text or "").strip()
                    if t:
                        rel.append(t)
            parent = (p.findtext("parent_section") or "").strip()
            pages_map[pid] = {
                "id": pid, "title": title, "description": desc, "importance": imp,
                "relevant_files": rf, "related_pages": rel, "parent_section": parent
            }

    meta = {
        "title": (root.findtext("title") or "").strip(),
        "description": (root.findtext("description") or "").strip(),
    }
    return {"meta": meta, "sections": sections, "pages": pages_map}

# ========================= Mermaid =========================

MERMAID_BLOCK_RE = re.compile(r"```mermaid\s+([\s\S]*?)```", re.IGNORECASE)

def extract_mermaid_blocks(markdown_text: str) -> List[str]:
    return [m.group(1).strip() for m in MERMAID_BLOCK_RE.finditer(markdown_text or "")]

def render_mermaid(mermaid_code: str, height: int = 500):
    """Render Mermaid diagrams with no extra pip deps (uses unpkg)."""
    import streamlit_mermaid as stmd
    safe = (mermaid_code or "").replace("\\", "\\\\").replace("`", "\\`")
    cid = f"mermaid-{uuid.uuid4().hex}"

    stmd.st_mermaid(safe, key = cid)
# ========================= Embeddings IO =========================

def load_embeddings_bundle(emb_dir: str) -> Dict[str, Any]:
    """
    Expects:
      - units.parquet  (or units.parquet.jsonl.gz)
      - file.index, file_ids.json
      - symbol.index, symbol_ids.json
    """
    units_path = os.path.join(emb_dir, "units.parquet")
    units_alt = os.path.join(emb_dir, "units.parquet.jsonl.gz")
    if os.path.isfile(units_path):
        df = pd.read_parquet(units_path)
    elif os.path.isfile(units_alt):
        rows = []
        with gzip.open(units_alt, "rt", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame.from_records(rows)
    else:
        raise FileNotFoundError("units.parquet not found in embeddings dir")

    try:
        import faiss as _faiss  # noqa
    except ImportError:
        raise RuntimeError("faiss-cpu is required. Install: pip install faiss-cpu")

    file_index = _faiss.read_index(os.path.join(emb_dir, "file.index"))
    symbol_index = _faiss.read_index(os.path.join(emb_dir, "symbol.index"))
    file_ids = json.loads(read_text(os.path.join(emb_dir, "file_ids.json")))
    symbol_ids = json.loads(read_text(os.path.join(emb_dir, "symbol_ids.json")))
    return {"df": df, "file_index": file_index, "symbol_index": symbol_index, "file_ids": file_ids, "symbol_ids": symbol_ids}

def embed_query(texts: List[str]) -> np.ndarray:
    try:
        from qgenie import QGenieClient
    except ImportError:
        raise RuntimeError("Install qgenie: pip install qgenie-sdk -i https://devpi.qualcomm.com/qcom/dev/+simple --trusted-host devpi.qualcomm.com")
    client = QGenieClient()
    embedding_response = client.embeddings(texts)
    embeddings = [item.embedding for item in embedding_response.data]
    return np.asarray(embeddings, dtype=np.float32)

def section_normalize(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"[\s\-_]+", " ", t)
    t = t.replace("&", "and")
    return t

# ---------- Per-section lexical hints ----------
SECTION_HINTS = {
    "system architecture": [
        "architecture", "design", "diagram", "docs/architecture", "docs/design",
        ".github/workflows", "dockerfile", "docker-compose", "helm", "k8s", "terraform", "ansible"
    ],
    "core features": ["core/", "models/", "transformers/", "qefficient/", "pipeline", "api", "feature"],
    "data management/flow": ["data/", "dataset", "db/", ".sql", "migrations", "etl", "pipeline"],
    "backend systems": ["api/", "/controllers/", "/services/", "/routes/", "server", "backend"],
    "model integration": ["modeling_", "inference", "training", "weights", "checkpoint", "load_model"],
    "deployment/infrastructure": ["dockerfile", "docker-compose", "helm", "k8s", "terraform", "ansible", ".github/workflows", "jenkinsfile", "cloudbuild"],
    "examples and notebooks": ["examples/", ".ipynb", "notebooks/"],
    "overview": ["readme", "docs/index", "docs/readme"],
    "extensibility and customization": ["plugin", "extension", "hooks", "interface", "adapter"],
}

def find_lexical_candidates(df: pd.DataFrame, keywords: List[str], max_files: int = 8, max_symbols: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (file_df_subset, symbol_df_subset) matched by substrings in file_path/symbol_name."""
    if not keywords:
        return df.head(0).copy(), df.head(0).copy()
    kw = [k.lower() for k in keywords if k]
    mask_files = (df["level"] == "file")
    mask_syms = (df["level"] == "symbol")

    def any_kw_in_path(p: str) -> bool:
        pl = (p or "").lower()
        return any(k in pl for k in kw)

    def any_kw_in_sym(r) -> bool:
        cand = [(r.get("file_path") or ""), (r.get("symbol_name") or ""), (r.get("signature") or "")]
        cand = " ".join([c.lower() for c in cand])
        return any(k in cand for k in kw)

    f_hits = df[mask_files].copy()
    f_hits = f_hits[f_hits["file_path"].apply(any_kw_in_path)]
    s_hits = df[mask_syms].copy()
    s_hits = s_hits[s_hits.apply(any_kw_in_sym, axis=1)]

    # simple ranking: shorter paths first (likely core), then have summary
    if len(f_hits):
        f_hits = f_hits.assign(_len=f_hits["file_path"].str.len(),
                               _has_sum=f_hits["summary"].fillna("").str.len() > 0)
        f_hits = f_hits.sort_values(by=["_has_sum", "_len"], ascending=[False, True]).drop(columns=["_len","_has_sum"]).head(max_files)
    if len(s_hits):
        s_hits = s_hits.assign(_has_sum=s_hits["summary"].fillna("").str.len() > 0)
        s_hits = s_hits.sort_values(by=["_has_sum"], ascending=[False]).drop(columns=["_has_sum"]).head(max_symbols)
    return f_hits, s_hits

def search_hybrid_plus(
    query: str,
    section_title: str,
    emb: Dict[str, Any],
    topk_file: int,
    topk_symbol: int,
    extra_file: int = 6,
    extra_symbol: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hybrid FAISS retrieval + per-section lexical boosters; returns (file_hits, sym_hits)."""
    import faiss  # noqa

    # Vector search
    qvec = embed_query([query])[0].reshape(1, -1)
    Df, If = emb["file_index"].search(qvec, topk_file)
    Ds, Is = emb["symbol_index"].search(qvec, topk_symbol)
    fids = [emb["file_ids"][i] for i in If[0]]
    sids = [emb["symbol_ids"][i] for i in Is[0]]
    df = emb["df"].set_index("uid")
    file_hits = df.loc[fids].reset_index().assign(score=Df[0])
    sym_hits = df.loc[sids].reset_index().assign(score=Ds[0])

    # Lexical boosters by section
    hints = SECTION_HINTS.get(section_normalize(section_title), [])
    f_boost, s_boost = find_lexical_candidates(df.reset_index(), hints, max_files=extra_file, max_symbols=extra_symbol)

    # Combine and drop duplicates by uid
    def combine(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        if len(b) == 0: return a
        both = pd.concat([a, b], ignore_index=True)
        both = both.drop_duplicates(subset=["uid"], keep="first")
        return both

    file_hits = combine(file_hits, f_boost)
    sym_hits = combine(sym_hits, s_boost)

    return file_hits, sym_hits

# ========================= Prompt Builders =========================

SECTION_TO_MERMAID = {
    "system architecture": ["flowchart", "graph TD", "graph LR"],
    "backend systems": ["sequenceDiagram", "flowchart"],
    "model integration": ["sequenceDiagram", "classDiagram"],
    "data management/flow": ["erDiagram", "flowchart"],
    "deployment/infrastructure": ["flowchart", "graph LR", "graph TD"],
    "examples and notebooks": ["flowchart"],
    "overview": ["flowchart"],
    "extensibility and customization": ["classDiagram", "flowchart"],
}

ALLOWED_SECTIONS = [
    "overview",
    "system architecture",
    "core features",
    "data management/flow",
    "backend systems",
    "model integration",
    "deployment/infrastructure",
    "examples and notebooks",
    "extensibility and customization",
]

FORBIDDEN_SECTIONS = [
    "text examples", "c++ examples", "python examples", "random scripts", "misc", "playground"
]

def build_context_pack(file_hits: pd.DataFrame, sym_hits: pd.DataFrame,
                       max_units: int = 16, max_code_chars: int = 1200) -> str:
    """Compact context from retrieved rows."""
    rows = []
    # Prefer symbols; fill with files
    for _, r in sym_hits.head(max_units).iterrows():
        rows.append({
            "kind": f"symbol::{r.get('symbol_type') or ''}",
            "path": r.get("file_path"),
            "name": r.get("symbol_name") or r.get("signature") or "",
            "summary": (r.get("summary") or "")[:800],
            "docstring": (r.get("docstring") or "")[:800],
            "code": (r.get("code") or "")[:max_code_chars]
        })
    remain = max(0, max_units - len(rows))
    if remain > 0:
        for _, r in file_hits.head(remain).iterrows():
            rows.append({
                "kind": "file",
                "path": r.get("file_path"),
                "name": "",
                "summary": (r.get("summary") or "")[:800],
                "docstring": "",
                "code": (r.get("code") or "")[:max_code_chars]
            })
    return json.dumps(rows, ensure_ascii=False, indent=2)

def build_page_prompt(
    repo_id: str,
    section_title: str,
    page_spec: Dict[str, Any],
    file_hits: pd.DataFrame,
    sym_hits: pd.DataFrame,
    lang_text: str,
    readme_ok: bool,
    min_files_per_page: int = 2,
    max_files_per_page: int = 5,
    max_units: int = 16,
    max_code_chars: int = 1200,
) -> Tuple[str, str]:
    """Return (draft_prompt, refine_prompt)."""
    sec_norm = section_normalize(section_title)
    mermaid_pref = SECTION_TO_MERMAID.get(sec_norm, ["flowchart"])
    context_block = build_context_pack(file_hits, sym_hits, max_units=max_units, max_code_chars=max_code_chars)

    xml_files_hint = page_spec.get("relevant_files") or []
    xml_files_hint_text = json.dumps(xml_files_hint, ensure_ascii=False, indent=2) if xml_files_hint else "[]"

    avoid_readme = "EXCEPT in 'Overview' or 'Examples and Notebooks'." if not readme_ok else "Allowed for this page."

    draft = f"""You are a senior technical writer and software architect documenting the repository '{repo_id}'.

PAGE SPEC:
- Section: {section_title}
- Page ID: {page_spec.get('id')}
- Title: {page_spec.get('title')}
- Description: {page_spec.get('description')}
- Importance: {page_spec.get('importance')}

ALLOWED vs FORBIDDEN SECTION TITLES (context; do not create new sections):
- Allowed: {ALLOWED_SECTIONS}
- Forbidden: {FORBIDDEN_SECTIONS}

FILE GROUNDEDNESS (HARD RULES):
- Use ONLY file paths from the context units below; do NOT invent paths.
- Include **{min_files_per_page}–{max_files_per_page}** exact file paths (verbatim) in a "References" section.
- README-like files (README, README.md/.rst/.txt) are {avoid_readme}
- Prefer code/config files that truly support the page topic.

MERMAID DIAGRAMS:
- Include at least one Mermaid diagram suitable for this section.
- Recommended types for '{section_title}': {mermaid_pref}
- Valid: sequenceDiagram, classDiagram, stateDiagram-v2, flowchart, erDiagram, graph LR, graph TD.
- Use fenced code blocks:
```mermaid
<diagram here>
```
- Add a short caption below each diagram.

OUTPUT FORMAT (STRICT):
- Return **ONLY Markdown**. No front-matter, no commentary.
- Suggested headings:
  # {page_spec.get('title')}
  ## Overview
  ## Key Components / Concepts
  ## How it Works
  ## Example(s)
  ## Diagram(s)
  ## References (with exact file paths)
- Be concise yet complete; avoid generic filler.

CONTEXT UNITS (use these only; do not invent file paths):
{context_block}

XML 'relevant_files' hints (optional):
{xml_files_hint_text}
"""

    refine = f"""You are a meticulous documentation editor.

Refine the Markdown for page '{page_spec.get('title')}' under '{section_title}' for repo '{repo_id}'.

CHECKLIST (MUST):
1) Content is grounded in file paths from the provided context.
2) "References" contains **{min_files_per_page}–{max_files_per_page}** exact file paths used in the write-up.
3) Mermaid diagram(s) exist and use suitable type(s) for this section (e.g., {mermaid_pref}); fix syntax if needed.
4) Headings present: # Title, ## Overview, ## Key Components / Concepts, ## How it Works, ## Example(s), ## Diagram(s), ## References.
5) No generic filler; precise explanations aligned with the code/summaries.
6) Language: {lang_text}. No extra commentary.

Return ONLY the final Markdown (no extra text).
ORIGINAL DRAFT:
"""
    return draft, refine

# ========================= QGenie =========================

def qgenie_chat(prompt: str, system: Optional[str] = None) -> str:
    """
    Sends a chat prompt to QGenie and returns the response.
    """
    try:
        from qgenie import QGenieClient, ChatMessage
        client = QGenieClient()
    except ImportError:
        raise RuntimeError(
            "qgenie not installed: pip install qgenie-sdk -i https://devpi.qualcomm.com/qcom/dev/+simple --trusted-host devpi.qualcomm.com"
        )

    messages = []
    if system:
        messages.append(ChatMessage(role="system", content=system))
    messages.append(ChatMessage(role="user", content=prompt))

    response = client.chat(messages=messages)
    return (getattr(response, "first_content", None) or str(response) or "").strip()

# ========================= Repo Meta from Wiki Filename =========================

def infer_graph_id_from_wiki_path(wiki_path: str) -> str:
    # Expect: <owner>__<repo>__<branch>__<graphid>__wiki.xml
    base = os.path.basename(wiki_path or "")
    if base.endswith("__wiki.xml"):
        core = base[:-len("__wiki.xml")]
        parts = core.split("__")
        if len(parts) >= 4:
            return parts[-1]
    parent = os.path.basename(os.path.dirname(wiki_path or ""))
    return parent or "graph"

def parse_repo_from_wiki_path(wiki_path: str) -> Tuple[str, str, str, str]:
    """Return (owner, repo, branch, graph_id) if encoded in filename; else blanks."""
    base = os.path.basename(wiki_path or "")
    owner = repo = branch = ""
    graph_id = infer_graph_id_from_wiki_path(wiki_path)
    if base.endswith("__wiki.xml"):
        core = base[:-len("__wiki.xml")]
        parts = core.split("__")
        if len(parts) >= 4:
            owner, repo, branch = parts[0], parts[1], parts[2]
    return owner, repo, branch, graph_id

# ========================= Post-processing Helpers =========================

def linkify_references(md: str, owner: str, repo: str, branch: str, subpath: str = "") -> str:
    """
    Convert file paths in '## References' into markdown links to GitHub blob URLs.
    Assumes references are listed as bullet points or inline code paths.
    - Backticked paths become clickable: `path`
    - If list items are plain text, they become clickable backticked links too.
    """
    if not (owner and repo and branch) or not md:
        return md
    base = f"https://github.com/{owner}/{repo}/blob/{branch}/"
    if subpath:
        sp = subpath.strip("/").rstrip("/")
        if sp:
            base += sp + "/"

    lines = md.splitlines()
    out = []
    in_refs = False
    for i, line in enumerate(lines):
        hdr = line.strip().lower()
        if hdr.startswith("## references"):
            in_refs = True
            out.append(line)
            continue
        if in_refs:
            # Exit references when next header appears
            if hdr.startswith("## ") and i != 0:
                in_refs = False
                out.append(line)
                continue

            replaced = line

            # 1) Linkify any backticked paths
            m_code = re.findall(r"`([^`]+)`", line)
            if m_code:
                for p in m_code:
                    p_clean = p.strip()
                    if not p_clean:
                        continue
                    url = base + p_clean.lstrip("/")
                    replaced = replaced.replace(f"`{p}`", f"`{p}`")
                out.append(replaced)
                continue

            # 2) Otherwise, try to detect a list item token and linkify it if it looks like a path
            m = re.match(r"^\s*([-*]|\d+\.)\s+(.+)$", line)
            if m:
                p = m.group(2).strip()
                # crude heuristic: has '/' or a dot ext in basename
                if ("/" in p) or ("." in os.path.basename(p)):
                    url = base + p.lstrip("/")
                    replaced = line.replace(p, f"`{p}`")
            out.append(replaced)
        else:
            out.append(line)
    return "\n".join(out)

def insert_mermaid_fallback(md: str, diagram_code: str) -> str:
    """Append or inject a Mermaid diagram block."""
    block = f"\n```mermaid\n{diagram_code.strip()}\n```\n\n*Diagram: auto-generated fallback.*\n"
    # Insert under "## Diagram(s)" if present, else append
    if "## Diagram(s)" in md:
        return md.replace("## Diagram(s)", "## Diagram(s)\n" + block, 1)
    return md + "\n\n## Diagram(s)\n" + block

# ========================= Streamlit App =========================

st.set_page_config(page_title="Page DocGen (Embeddings + QGenie + Mermaid)", layout="wide")
st.title("🧾 Page-level Documentation Generator (RAG + QGenie + Mermaid)")

# --- Persist generated outputs across reruns ---
if "gen_results" not in st.session_state:
    st.session_state.gen_results = {}  # pid -> final_md
if "gen_paths" not in st.session_state:
    st.session_state.gen_paths = {}    # pid -> file_path

with st.sidebar:
    st.header("Inputs")
    wiki_xml_path = st.text_input("Wiki XML path", help="Path to the wiki_structure XML file.")
    # Auto-detect embeddings folder from graph_id
    suggested_emb_dir = ""
    suggested_graph_id = ""
    if wiki_xml_path:
        suggested_graph_id = infer_graph_id_from_wiki_path(wiki_xml_path)
        if suggested_graph_id:
            suggested_emb_dir = os.path.join(".cache", "embeddings", suggested_graph_id)
    embed_dir = st.text_input("Embeddings folder", value=suggested_emb_dir, help="E.g., .cache/embeddings/<graph_id>")
    lang_text = st.text_input("Language for docs", value="English")

    st.divider()
    st.header("Repo link settings (for clickable References)")
    owner, repo, branch, gid = parse_repo_from_wiki_path(wiki_xml_path) if wiki_xml_path else ("","","","")
    owner = st.text_input("GitHub owner", value=owner)
    repo = st.text_input("GitHub repo", value=repo)
    branch = st.text_input("GitHub branch", value=branch or "main")
    subpath = st.text_input("Subpath within repo (optional)", value="")

    st.divider()
    st.header("Retrieval")
    topk_symbol = st.number_input("Top-K symbols (FAISS)", min_value=3, max_value=50, value=12, step=1)
    topk_file = st.number_input("Top-K files (FAISS)", min_value=2, max_value=50, value=8, step=1)
    max_units = st.number_input("Max units in context", min_value=6, max_value=50, value=16, step=1)
    max_code_chars = st.number_input("Max code chars per unit", min_value=300, max_value=4000, value=1200, step=100)

    st.divider()
    st.header("Generation")
    min_refs = st.number_input("Min file refs/page", min_value=1, max_value=8, value=2)
    max_refs = st.number_input("Max file refs/page", min_value=2, max_value=10, value=5)
    include_overview_readme = st.checkbox("Allow README refs in Overview/Examples", value=True)
    diagram_fallback = st.checkbox("Enable diagram-only fallback if none found", value=True)

    st.divider()
    st.header("Caching")
    use_cache = st.checkbox("Use cached outputs if available", value=True)
    clear_cache = st.checkbox("Force regenerate (ignore cache)", value=False)

    st.divider()
    show_draft_prompt = st.checkbox("Show draft prompt", value=False)
    show_refine_prompt = st.checkbox("Show refine prompt", value=False)

    st.divider()
    generate_btn = st.button("Generate Selected Pages", type="primary", use_container_width=True)
    generate_all_btn = st.button("Generate ALL Pages", type="secondary", use_container_width=True)

# Load inputs
wiki = None
emb = None
pages_order: List[str] = []
sections_by_id: Dict[str, str] = {}
if wiki_xml_path and os.path.isfile(wiki_xml_path):
    try:
        wiki = load_wiki_xml(wiki_xml_path)
        # section id -> title
        for s in wiki.get("sections", []):
            sections_by_id[s["id"]] = s["title"]
        # stable page order by sections
        seen = set()
        for s in wiki.get("sections", []):
            for pid in s.get("page_refs", []):
                if pid in wiki["pages"] and pid not in seen:
                    pages_order.append(pid); seen.add(pid)
        # orphans
        for pid in wiki.get("pages", {}).keys():
            if pid not in seen:
                pages_order.append(pid)
    except Exception as e:
        st.error(f"Failed to parse wiki XML: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

if embed_dir and os.path.isdir(embed_dir):
    try:
        emb = load_embeddings_bundle(embed_dir)
    except Exception as e:
        st.error(f"Failed to load embeddings bundle: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

colA, colB = st.columns([1.5, 1])
with colA:
    st.subheader("Pages")
    selected_pages = []
    if wiki:
        options = []
        for pid in pages_order:
            p = wiki["pages"][pid]
            sec_title = sections_by_id.get(p.get("parent_section", ""), "(Unknown Section)")
            options.append((pid, f"[{pid}] {p.get('title')} — {sec_title}"))
        if not options:
            st.info("No pages found in XML.")
        else:
            idxs = st.multiselect(
                "Select pages to generate",
                options=list(range(len(options))),
                default=list(range(min(3, len(options)))),
                format_func=lambda i: options[i][1],
            )
            selected_pages = [options[i][0] for i in idxs]
with colB:
    st.subheader("Output folder")
    inferred_gid = infer_graph_id_from_wiki_path(wiki_xml_path) if wiki_xml_path else "graph"
    out_dir = ensure_dir(os.path.join(".cache", "wiki_pages", inferred_gid))
    cache_dir = ensure_dir(os.path.join(out_dir, ".cache"))
    st.code(out_dir, language="bash")

# ========================= Generation Core =========================

def build_cache_key(
    page_spec: Dict[str, Any],
    sec_title: str,
    retrieval_knobs: Dict[str, Any],
    gen_knobs: Dict[str, Any],
    ctx_unit_ids: List[str],
    version: str = "v2"
) -> str:
    payload = {
        "v": version,
        "page": {k: page_spec.get(k) for k in ["id","title","description","importance","parent_section"]},
        "section": sec_title,
        "retrieval": retrieval_knobs,
        "generation": gen_knobs,
        "ctx_ids": ctx_unit_ids[:64],  # cap
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]

def page_cache_paths(cache_dir: str, page_id: str, key: str) -> Tuple[str, str]:
    return os.path.join(cache_dir, f"{page_id}__{key}__draft.md"), os.path.join(cache_dir, f"{page_id}__{key}__final.md")

def context_ids_from_hits(file_hits: pd.DataFrame, sym_hits: pd.DataFrame, max_units: int) -> List[str]:
    ids = []
    if len(sym_hits):
        ids.extend(sym_hits["uid"].tolist()[:max_units])
    if len(file_hits):
        # fill until max_units*2 just for stability
        remain = max(0, max_units*2 - len(ids))
        ids.extend(file_hits["uid"].tolist()[:remain])
    return ids

def generate_one_page(
    pid: str,
    wiki: Dict[str, Any],
    emb: Dict[str, Any],
    lang_text: str,
    retrieval_knobs: Dict[str, Any],
    gen_knobs: Dict[str, Any],
    out_dir: str,
    cache_dir: str,
    owner: str, repo: str, branch: str, subpath: str,
    show_prompts: Tuple[bool,bool],
    diagram_fallback: bool,
    use_cache: bool,
    clear_cache: bool,
) -> Tuple[str, str]:
    """
    Returns (page_path_written, final_markdown)
    """
    p = wiki["pages"][pid]
    sec_title = sections_by_id.get(p.get("parent_section", ""), "(Unknown Section)")
    repo_title = wiki["meta"].get("title") or f"{owner}/{repo}" or "Repository"

    # Build per-section retrieval query (augmented)
    sec_norm = section_normalize(sec_title)
    key_terms = SECTION_HINTS.get(sec_norm, [])
    # Start with page title + section + description + boosted terms
    query = f"{p.get('title')} — {sec_title}. {p.get('description')}. " + " ".join(key_terms[:8])

    # Retrieve hybrid + lexical boosters
    file_hits, sym_hits = search_hybrid_plus(
        query=query,
        section_title=sec_title,
        emb=emb,
        topk_file=int(retrieval_knobs["topk_file"]),
        topk_symbol=int(retrieval_knobs["topk_symbol"]),
        extra_file=6, extra_symbol=10
    )

    # Build prompts
    readme_ok = gen_knobs["include_overview_readme"] and (sec_norm in ("overview", "examples and notebooks"))
    draft_prompt, refine_prompt = build_page_prompt(
        repo_id=repo_title,
        section_title=sec_title,
        page_spec=p,
        file_hits=file_hits,
        sym_hits=sym_hits,
        lang_text=lang_text,
        readme_ok=readme_ok,
        min_files_per_page=int(gen_knobs["min_refs"]),
        max_files_per_page=int(gen_knobs["max_refs"]),
        max_units=int(retrieval_knobs["max_units"]),
        max_code_chars=int(retrieval_knobs["max_code_chars"]),
    )

    if show_prompts[0] or show_prompts[1]:
        with st.expander(f"🔧 {pid} — Prompts", expanded=False):
            if show_prompts[0]:
                st.markdown("**Draft prompt**")
                st.code(draft_prompt)
            if show_prompts[1]:
                st.markdown("**Refine prompt**")
                st.code(refine_prompt)

    # Caching
    ctx_ids = context_ids_from_hits(file_hits, sym_hits, int(retrieval_knobs["max_units"]))
    cache_key = build_cache_key(
        page_spec=p,
        sec_title=sec_title,
        retrieval_knobs=retrieval_knobs,
        gen_knobs=gen_knobs,
        ctx_unit_ids=ctx_ids,
        version="v2"
    )
    draft_cache_path, final_cache_path = page_cache_paths(cache_dir, pid, cache_key)

    final_md = None
    if use_cache and (not clear_cache) and os.path.isfile(final_cache_path):
        final_md = read_text(final_cache_path)

    if final_md is None:
        # QGenie: draft -> refine
        system_msg = ("Follow the user's instructions EXACTLY. Output ONLY Markdown; "
                      "include required Mermaid diagrams; obey file selection constraints; "
                      "no extra commentary.")
        draft_md = qgenie_chat(draft_prompt, system=system_msg)
        write_text(draft_cache_path, draft_md)

        final_md = qgenie_chat(refine_prompt + "\n\n" + draft_md, system=system_msg)
        # Diagram-only fallback (if none)
        if diagram_fallback:
            diags = extract_mermaid_blocks(final_md)
            if not diags:
                # Short fallback prompt (sequenceDiagram suggested)
                fallback = f"""Generate only a Mermaid diagram for the page '{p.get('title')}' ({sec_title}).
Use 'sequenceDiagram' unless a class or flowchart is clearly better; return a single Mermaid fenced block and nothing else.

Context units (for accurate names & calls):
{build_context_pack(file_hits, sym_hits, max_units=int(retrieval_knobs["max_units"]), max_code_chars=600)}
"""
                fb_md = qgenie_chat(fallback,  system="Return ONLY a mermaid fenced block.")
                # Extract fence or wrap
                code = ""
                m = re.search(r"```mermaid\s+([\s\S]*?)```", fb_md, re.IGNORECASE)
                if m:
                    code = m.group(1).strip()
                else:
                    # If model returned raw mermaid without fence, use as-is
                    code = fb_md.strip().strip("`")
                if code:
                    final_md = insert_mermaid_fallback(final_md, code)

        # Linkify references to GitHub
        final_md = linkify_references(final_md, owner, repo, branch, subpath)
        # Save final cache
        write_text(final_cache_path, final_md)

    # Save page output (stable path)
    page_path = os.path.join(out_dir, f"{pid}.md")
    write_text(page_path, final_md)
    return page_path, final_md

# ========================= Run & Preview =========================

def run_generation(page_ids: List[str]):
    if not wiki or not emb:
        st.error("Load both Wiki XML and Embeddings first.")
        return
    if not page_ids:
        st.warning("Select at least one page.")
        return

    retrieval_knobs = {
        "topk_symbol": int(topk_symbol),
        "topk_file": int(topk_file),
        "max_units": int(max_units),
        "max_code_chars": int(max_code_chars),
    }
    gen_knobs = {
        "min_refs": int(min_refs),
        "max_refs": int(max_refs),
        "include_overview_readme": bool(include_overview_readme),
    }

    prog = st.progress(0.0, text="Starting...")
    n = len(page_ids)
    for i, pid in enumerate(page_ids, start=1):
        try:
            with st.status(f"Generating page **{pid}**...", expanded=False):
                page_path, final_md = generate_one_page(
                    pid=pid,
                    wiki=wiki,
                    emb=emb,
                    lang_text=lang_text,
                    retrieval_knobs=retrieval_knobs,
                    gen_knobs=gen_knobs,
                    out_dir=out_dir,
                    cache_dir=cache_dir,
                    owner=owner, repo=repo, branch=branch, subpath=subpath,
                    show_prompts=(show_draft_prompt, show_refine_prompt),
                    diagram_fallback=diagram_fallback,
                    use_cache=use_cache,
                    clear_cache=clear_cache,
                )
                # Persist results for cross-rerun rendering and downloads
                st.session_state.gen_results[pid] = final_md
                st.session_state.gen_paths[pid] = page_path

                st.success(f"Generated {pid} → {page_path}")
        except Exception as e:
            st.error(f"Failed to generate page {pid}: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
        finally:
            prog.progress(i / max(1, n), text=f"{i}/{n} done")

# Trigger generation
if generate_btn:
    run_generation(selected_pages)

if generate_all_btn:
    run_generation(pages_order)

# ========================= Preview Generated Pages =========================
st.divider()
st.header("📄 Preview Generated Pages")

if not st.session_state.gen_results:
    st.info("No generated pages yet. Use the buttons in the sidebar to generate pages.")
else:
    # Build options for selection
    _opts = []
    for pid, md in st.session_state.gen_results.items():
        title = wiki["pages"].get(pid, {}).get("title", pid) if wiki else pid
        sec_title = "(Unknown Section)"
        if wiki:
            sec_title = sections_by_id.get(wiki["pages"].get(pid, {}).get("parent_section", ""), "(Unknown Section)")
        _opts.append((pid, f"[{pid}] {title} — {sec_title}"))
    # stable list
    pid_to_label = dict(_opts)
    ordered_pids = [pid for pid, _ in _opts if pid in st.session_state.gen_results]

    preview_idx = st.multiselect(
        "Select pages to preview",
        options=list(range(len(ordered_pids))),
        default=list(range(min(3, len(ordered_pids)))),
        format_func=lambda i: pid_to_label[ordered_pids[i]],
    )
    preview_pids = [ordered_pids[i] for i in preview_idx]

    render_diagrams = st.checkbox("Render Mermaid diagrams in preview", value=True)

    for pid in preview_pids:
        final_md = st.session_state.gen_results[pid]
        title = wiki["pages"].get(pid, {}).get("title", pid) if wiki else pid
        page_path = st.session_state.gen_paths.get(pid, os.path.join(out_dir, f"{pid}.md"))

        st.markdown("---")
        st.subheader(f"{title}  \n`{pid}.md`")
        # Render the markdown content
        st.markdown(final_md, unsafe_allow_html=False)

        # Render Mermaid diagrams below the markdown if enabled
        diags = extract_mermaid_blocks(final_md)
        if render_diagrams:
            if diags:
                st.info(f"Rendering {len(diags)} Mermaid diagram(s)...")
                for j, d in enumerate(diags, start=1):
                    st.markdown(f"**Diagram {j}**")
                    render_mermaid(d, height=520)
            else:
                st.warning("No Mermaid diagrams detected in the output.")

        # Per-page download
        st.download_button(
            f"Download {pid}.md",
            data=final_md,
            file_name=f"{pid}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ========================= Download ZIP =========================
if st.session_state.gen_results:
    st.markdown("### Download all generated pages")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pid, md in st.session_state.gen_results.items():
            zf.writestr(f"{pid}.md", md)
    st.download_button(
        "Download ZIP",
        data=buf.getvalue(),
        file_name="wiki_pages.zip",
        mime="application/zip",
        use_container_width=True,
    )
