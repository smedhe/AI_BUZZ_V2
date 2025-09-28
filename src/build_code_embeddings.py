
"""
Build hybrid code embeddings (file-level + function/class-level) for a GitHub repository.

Features:
- Download GitHub repo as ZIP, parse polyglot code.
- Extract file-level units and symbol-level (function/class) units.
- (Optional) Summarize each unit with QGenie for better NL alignment.
- Embed with sentence-transformers; build FAISS indices.
- Save all artifacts under .cache/embeddings/<graph_id>/.

Usage:
  # Basic (no summarization)
  python build_code_embeddings.py \
    --github-url https://github.com/pallets/flask \
    --embed-model sentence-transformers/all-MiniLM-L6-v2

  # With QGenie summarization (better retrieval)
  python build_code_embeddings.py \
    --github-url https://github.com/pallets/flask \
    --embed-model intfloat/e5-base-v2 \
    --summarize \
    --qgenie-model qwen2.5-14b-1m \
    --summarize-max 200

  # Query the built indices (hybrid retrieval)
  python build_code_embeddings.py \
    --query "How is request routing implemented?" \
    --github-url https://github.com/pallets/flask \
    --embed-model sentence-transformers/all-MiniLM-L6-v2 \
    --topk 5
"""

from __future__ import annotations

import argparse
import ast
import gzip
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Optional heavy deps are imported lazily:
# - sentence_transformers
# - faiss
# - qgenie

# ===================== Repo Download / IO =====================

SKIP_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "build", "dist", "node_modules",
    ".mypy_cache", ".pytest_cache", "target", "out", ".gradle", ".idea"
}
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2MB per file safety

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True); return path

def parse_github_url(url: str) -> Dict[str, str]:
    url = url.strip()
    url = re.sub(r"\.git$", "", url)
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)(?:/(tree|blob)/([^/]+)(/.*)?)?$", url)
    if not m:
        raise ValueError("Expect https://github.com/<owner>/<repo>[...].")
    owner, repo, kind, branch, subpath = m.groups()
    subpath = (subpath or "").lstrip("/")
    return {"owner": owner, "repo": repo, "branch": branch or "", "subpath": subpath}

def http_get_json(url: str, token: Optional[str] = None):
    from urllib.request import Request, urlopen
    headers = {"User-Agent": "code-embedder"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        import json as _json
        return _json.load(io.TextIOWrapper(resp, encoding="utf-8"))

def get_default_branch(owner: str, repo: str, token: Optional[str] = None) -> str:
    try:
        info = http_get_json(f"https://api.github.com/repos/{owner}/{repo}", token=token)
        if info.get("default_branch"):
            return info["default_branch"]
    except Exception:
        pass
    return "main"

def can_download_zip(owner: str, repo: str, branch: str) -> bool:
    from urllib.request import Request, urlopen
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    try:
        req = Request(url, headers={"User-Agent": "code-embedder"})
        with urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False

def download_repo_zip(owner: str, repo: str, branch: str) -> bytes:
    from urllib.request import Request, urlopen
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    req = Request(url, headers={"User-Agent": "code-embedder"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()

def unzip_to_temp(zip_bytes: bytes) -> str:
    tmpdir = tempfile.mkdtemp(prefix="ghrepo_")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmpdir)
    # pick biggest dir by file count
    entries = [os.path.join(tmpdir, d) for d in os.listdir(tmpdir)]
    dirs = [p for p in entries if os.path.isdir(p)]
    if not dirs:
        raise RuntimeError("Unexpected ZIP structure.")
    return max(dirs, key=lambda p: sum(len(files) for _, _, files in os.walk(p)))

def iter_repo_files(root_dir: str, subpath: Optional[str] = None, max_files: Optional[int] = None):
    base = os.path.join(root_dir, subpath) if subpath else root_dir
    count = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            abs_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(abs_path, root_dir).replace("\\", "/")
            yield abs_path, rel_path
            # count += 1
            # if max_files and count >= max_files:
            #     return

def read_text_file(path: str) -> Optional[str]:
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return None
        with open(path, "rb") as f:
            data = f.read()
        if b"\x00" in data:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return None

def fingerprint(owner: str, repo: str, branch: str, subpath: Optional[str], url: str) -> str:
    payload = f"{owner}/{repo}@{branch}:{subpath or ''}|{url}"
    import hashlib
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

# ===================== Language Helpers (polyglot) =====================

PY_EXTS   = {".py"}
JS_TS_EXTS= {".js",".jsx",".ts",".tsx",".mjs",".cjs"}
JAVA_EXTS = {".java"}
GO_EXTS   = {".go"}
C_EXTS    = {".c",".h"}
CPP_EXTS  = {".cc",".cpp",".cxx",".hpp",".hh",".hxx",".c++",".h++"}
RUST_EXTS = {".rs"}
RUBY_EXTS = {".rb"}
PHP_EXTS  = {".php"}
KT_EXTS   = {".kt",".kts"}

def detect_lang(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in PY_EXTS: return "python"
    if ext in JS_TS_EXTS: return "javascript"
    if ext in JAVA_EXTS: return "java"
    if ext in GO_EXTS: return "go"
    if ext in C_EXTS or ext in CPP_EXTS: return "cpp" if ext in CPP_EXTS else "c"
    if ext in RUST_EXTS: return "rust"
    if ext in RUBY_EXTS: return "ruby"
    if ext in PHP_EXTS: return "php"
    if ext in KT_EXTS: return "kotlin"
    return "text"

# ===================== Unit Extraction =====================

@dataclass
class Unit:
    uid: str                 # unique id (we'll create)
    level: str               # "file" | "symbol"
    file_path: str
    lang: str
    symbol_type: Optional[str] = None  # "function" | "class" | None
    symbol_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    code: Optional[str] = None
    summary: Optional[str] = None      # LLM summary (optional)

def extract_python_units(path: str, text: str) -> List[Unit]:
    units: List[Unit] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return units
    lines = text.splitlines()
    def get_segment(node):
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if lineno and end_lineno and 1 <= lineno <= len(lines) and 1 <= end_lineno <= len(lines):
            return "\n".join(lines[lineno-1:end_lineno]), lineno, end_lineno
        return None, None, None

    # Top-level file unit
    units.append(Unit(
        uid=f"file::{path}",
        level="file",
        file_path=path,
        lang="python",
        code=text,
        docstring=ast.get_docstring(tree) or None
    ))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            code, sl, el = get_segment(node)
            doc = ast.get_docstring(node)
            sig = f"{node.name}({', '.join(a.arg for a in node.args.args)})"
            units.append(Unit(
                uid=f"symbol::{path}::function::{node.name}::{sl or 0}",
                level="symbol",
                file_path=path, lang="python",
                symbol_type="function", symbol_name=node.name,
                start_line=sl, end_line=el,
                signature=sig, docstring=doc, code=code
            ))
        elif isinstance(node, ast.ClassDef):
            code, sl, el = get_segment(node)
            doc = ast.get_docstring(node)
            units.append(Unit(
                uid=f"symbol::{path}::class::{node.name}::{sl or 0}",
                level="symbol",
                file_path=path, lang="python",
                symbol_type="class", symbol_name=node.name,
                start_line=sl, end_line=el,
                signature=node.name, docstring=doc, code=code
            ))
    return units

# For other languages, we use heuristics to find function/class headers and capture a block window
_ident = r"[A-Za-z_][A-Za-z0-9_]*"
_js_ident = r"[A-Za-z_$][A-Za-z0-9_$]*"

def extract_js_ts_units(path: str, text: str) -> List[Unit]:
    units: List[Unit] = [Unit(uid=f"file::{path}", level="file", file_path=path, lang="javascript", code=text)]
    # classes
    for m in re.finditer(rf"\bclass\s+({_js_ident})\b", text):
        name = m.group(1)
        snippet = text[m.start(): m.start() + 2000]
        units.append(Unit(uid=f"symbol::{path}::class::{name}::{m.start()}",
                          level="symbol", file_path=path, lang="javascript",
                          symbol_type="class", symbol_name=name, start_line=None, end_line=None,
                          signature=name, code=snippet))
    # functions (common patterns)
    patterns = [
        rf"\bfunction\s+({_js_ident})\s*\(",
        rf"\bexport\s+function\s+({_js_ident})\s*\(",
        rf"\b(?:const|let|var)\s+({_js_ident})\s*=\s*function\b",
        rf"\b(?:const|let|var)\s+({_js_ident})\s*=\s*\(",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            name = m.group(1)
            snippet = text[m.start(): m.start() + 2000]
            units.append(Unit(uid=f"symbol::{path}::function::{name}::{m.start()}",
                              level="symbol", file_path=path, lang="javascript",
                              symbol_type="function", symbol_name=name, code=snippet))
    return units

def extract_simple_block_units(path: str, text: str, lang: str, func_re: str, class_re: Optional[str] = None) -> List[Unit]:
    units: List[Unit] = [Unit(uid=f"file::{path}", level="file", file_path=path, lang=lang, code=text)]
    # classes
    if class_re:
        for m in re.finditer(class_re, text):
            name = m.group(1)
            snippet = text[m.start(): m.start()+2000]
            units.append(Unit(uid=f"symbol::{path}::class::{name}::{m.start()}",
                              level="symbol", file_path=path, lang=lang,
                              symbol_type="class", symbol_name=name, code=snippet))
    # functions
    for m in re.finditer(func_re, text):
        name = m.group(1)
        snippet = text[m.start(): m.start()+2000]
        units.append(Unit(uid=f"symbol::{path}::function::{name}::{m.start()}",
                          level="symbol", file_path=path, lang=lang,
                          symbol_type="function", symbol_name=name, code=snippet))
    return units

def extract_units_for_file(path: str, text: str) -> List[Unit]:
    lang = detect_lang(path)
    if lang == "python":
        return extract_python_units(path, text)
    if lang == "javascript":
        return extract_js_ts_units(path, text)
    if lang == "java":
        return extract_simple_block_units(path, text, "java",
            func_re=rf"\b({_ident})\s*\([^;]*\)\s*\{{", class_re=rf"\bclass\s+({_ident})\b")
    if lang == "go":
        return extract_simple_block_units(path, text, "go",
            func_re=rf"\bfunc\s+(?:\([^)]+\)\s*)?({_ident})\s*\(", class_re=rf"\btype\s+({_ident})\s+struct\b")
    if lang in ("c","cpp"):
        return extract_simple_block_units(path, text, lang,
            func_re=rf"(?m)^[ \t]*(?:[_A-Za-z]\w*[\s\*\&\<\>:]+)+({_ident})\s*\([^;{{\)]*\)\s*\{{",
            class_re=rf"\b(class|struct)\s+({_ident})\b")  # note: group(2) would be name; we use simple capture
    if lang == "rust":
        return extract_simple_block_units(path, text, "rust",
            func_re=rf"\bfn\s+({_ident})\s*\(", class_re=rf"\b(struct|enum)\s+({_ident})\b")
    if lang == "ruby":
        return extract_simple_block_units(path, text, "ruby",
            func_re=rf"(?m)^\s*def\s+({_ident})\b", class_re=rf"\bclass\s+({_ident})\b")
    if lang == "php":
        return extract_simple_block_units(path, text, "php",
            func_re=rf"\bfunction\s+({_ident})\s*\(", class_re=rf"\bclass\s+({_ident})\b")
    if lang == "kotlin":
        return extract_simple_block_units(path, text, "kotlin",
            func_re=rf"\bfun\s+({_ident})\s*\(", class_re=rf"\b(class|object|interface)\s+({_ident})\b")
    # fallback file-only
    return [Unit(uid=f"file::{path}", level="file", file_path=path, lang=lang, code=text)]

# ===================== Summarization (QGenie) =====================

def summarize_units_with_qgenie(units: List[Unit], max_items: Optional[int] = None):
    try:
        from qgenie import QGenieClient, ChatMessage
    except ImportError:
        print("[WARN] qgenie not installed; skipping summarization.", file=sys.stderr)
        return
    client = QGenieClient()
    count = 0
    for u in units:
        if max_items and count >= max_items:
            break
        # Skip empty code
        snippet = u.code or ""
        if not snippet.strip():
            continue
        # Prefer docstring for python
        doc_hint = u.docstring or ""
        role = f"{u.symbol_type or 'file'}"
        name = u.symbol_name or os.path.basename(u.file_path)
        prompt = f"""Summarize the following {role} '{name}' for semantic search.

Return 2-4 sentences covering:
- Purpose and what it does,
- Inputs/outputs (if any),
- Key side-effects or important behaviors (if any).
Be concise, factual, and specific. Do not include code fences.

Docstring (may be empty):
{doc_hint}

Code (excerpt):
{snippet[:4000]}
"""
        try:
            resp = client.chat(messages=[ChatMessage(role="user", content=prompt)])
            u.summary = getattr(resp, "first_content", None) or str(resp)
            count += 1
        except Exception as e:
            print(f"[WARN] QGenie summarization failed for {u.uid}: {e}", file=sys.stderr)

# ===================== Embedding + FAISS =====================

from qgenie import QGenieClient

def embed_texts(texts: List[str], device: Optional[str] = None) -> np.ndarray:
    """
    Embeds a list of texts using QGenie embedding API.
    """
    client = QGenieClient()
    embedding_response = client.embeddings(texts)

    # Extract embeddings from response
    embeddings = [item.embedding for item in embedding_response.data]

    return np.asarray(embeddings, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS index using inner product (cosine similarity).
    """
    try:
        import faiss
    except ImportError:
        raise RuntimeError("Please install: pip install faiss-cpu")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine since we normalize)
    index.add(embeddings)
    return index


# ===================== Packing + Saving =====================

def units_to_dataframe(units: List[Unit]) -> pd.DataFrame:
    recs = []
    for u in units:
        recs.append({
            "uid": u.uid, "level": u.level, "file_path": u.file_path, "lang": u.lang,
            "symbol_type": u.symbol_type, "symbol_name": u.symbol_name,
            "start_line": u.start_line, "end_line": u.end_line,
            "signature": u.signature, "docstring": u.docstring,
            "summary": u.summary, "code": u.code
        })
    return pd.DataFrame.from_records(recs)

def default_output_dir(graph_id: str) -> str:
    return ensure_dir(os.path.join(".cache", "embeddings", graph_id))

def save_parquet(df: pd.DataFrame, path: str):
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Fallback to gzip JSONL if pyarrow isn't present
        with gzip.open(path + ".jsonl.gz", "wt", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

def save_index(index, ids: List[str], out_dir: str, prefix: str):
    # Save FAISS and ids
    index_path = os.path.join(out_dir, f"{prefix}.index")
    ids_path = os.path.join(out_dir, f"{prefix}_ids.json")
    try:
        import faiss
        faiss.write_index(index, index_path)
    except Exception as e:
        print(f"[WARN] Failed to save FAISS index: {e}", file=sys.stderr)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    return index_path, ids_path

# ===================== Query (hybrid) =====================

def hybrid_query(query: str, out_dir: str, topk: int = 5) -> List[Dict[str, Any]]:
    try:
        import faiss
    except ImportError:
        raise RuntimeError("pip install faiss-cpu")

    # Load ids
    file_ids_path = os.path.join(out_dir, "file_ids.json")
    symbol_ids_path = os.path.join(out_dir, "symbol_ids.json")
    with open(file_ids_path, "r", encoding="utf-8") as f:
        file_ids = json.load(f)
    with open(symbol_ids_path, "r", encoding="utf-8") as f:
        symbol_ids = json.load(f)

    # Load indices
    file_index = faiss.read_index(os.path.join(out_dir, "file.index"))
    symbol_index = faiss.read_index(os.path.join(out_dir, "symbol.index"))

    qvec = embed_texts([query])[0].reshape(1, -1)

    Df, If = file_index.search(qvec, topk)
    Ds, Is = symbol_index.search(qvec, topk)

    hits = []
    for score, idx in zip(Df[0].tolist(), If[0].tolist()):
        hits.append({"source": "file", "uid": file_ids[idx], "score": float(score)})
    for score, idx in zip(Ds[0].tolist(), Is[0].tolist()):
        hits.append({"source": "symbol", "uid": symbol_ids[idx], "score": float(score)})

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:topk]

# ===================== Main Pipeline =====================

def build_units_for_repo(github_url: str, token: Optional[str]) -> Tuple[str, Dict[str, Any], List[Unit]]:
    parts = parse_github_url(github_url)
    owner = parts["owner"]; repo = parts["repo"]
    branch = parts["branch"] or get_default_branch(owner, repo, token=token)
    subpath = parts["subpath"] or None
    gid = fingerprint(owner, repo, branch, subpath, github_url)

    # Download/unzip
    zip_bytes = download_repo_zip(owner, repo, branch)
    repo_root = unzip_to_temp(zip_bytes)

    # Extract units
    all_units: List[Unit] = []
    file_count = 0
    for abs_path, rel_path in iter_repo_files(repo_root, subpath=subpath):
        text = read_text_file(abs_path)
        if text is None:
            continue
        units = extract_units_for_file(rel_path, text)
        all_units.extend(units)
        file_count += 1

    meta = {
        "source_url": github_url,
        "owner": owner, "repo": repo, "branch": branch,
        "subpath": subpath or "", "created_at": now_iso(),
        "graph_id": gid, "totals": {"files_scanned": file_count, "units": len(all_units)}
    }

    # Cleanup tmp
    try:
        import shutil
        shutil.rmtree(os.path.dirname(repo_root))
    except Exception:
        pass

    return gid, meta, all_units

def main():
    p = argparse.ArgumentParser(description="Build hybrid code embeddings (file + symbol) for a GitHub repository.")
    p.add_argument("--github-url", required=True, help="Repo URL: https://github.com/<owner>/<repo>[/tree/<branch>/<subpath>]")
    p.add_argument("--token", default=None, help="GitHub token (optional) for better rate-limits")

    # Summarization
    p.add_argument("--summarize", action="store_true", help="Summarize units with QGenie (recommended)")
    p.add_argument("--summarize-max", type=int, default=400, help="Cap the number of units to summarize (cost/speed)")

    p.add_argument("--device", default=None, help="Embedding device (e.g., 'cpu' or 'cuda')")

    # Query
    p.add_argument("--query", default=None, help="Run a hybrid query against the built indices")
    p.add_argument("--topk", type=int, default=5)

    args = p.parse_args()

    # Build units
    gid, meta, units = build_units_for_repo(args.github_url, token=args.token)
    out_dir = default_output_dir(gid)
    ensure_dir(out_dir)

    print(f"[INFO] Repo graph_id: {gid}; units: {len(units)}")

    # Summarize (optional)
    if args.summarize:
        print("[INFO] Summarizing units with QGenie...")
        summarize_units_with_qgenie(units, max_items=args.summarize_max)

    # Pack dataframe
    df = units_to_dataframe(units)
    df_path = os.path.join(out_dir, "units.parquet")
    save_parquet(df, df_path)

    # Prepare embedding texts
    # - For files: prefer summary (if any) + path; else first 1500 chars code
    # - For symbols: prefer summary + signature + name + short code
    file_mask = (df["level"] == "file")
    sym_mask = (df["level"] == "symbol")

    file_df = df[file_mask].copy()
    sym_df = df[sym_mask].copy()

    def prep_file_text(row):
        base = row.get("summary") or ""
        if not base:
            code = (row.get("code") or "")[:1500]
            base = f"{row.get('file_path')}\n{code}"
        else:
            base = f"{row.get('file_path')}\n{base}"
        return base

    def prep_symbol_text(row):
        parts = []
        if row.get("summary"):
            parts.append(row["summary"])
        sig = row.get("signature") or row.get("symbol_name") or ""
        if sig:
            parts.append(sig)
        parts.append(row.get("file_path") or "")
        code = (row.get("code") or "")[:1500]
        if code:
            parts.append(code)
        return "\n".join([p for p in parts if p])

    file_texts = [prep_file_text(r) for _, r in file_df.iterrows()]
    sym_texts = [prep_symbol_text(r) for _, r in sym_df.iterrows()]

    # Embeddings
    print("[INFO] Embedding file-level texts...")
    file_vecs = embed_texts(file_texts,  device=args.device)
    print("[INFO] Embedding symbol-level texts...")
    sym_vecs = embed_texts(sym_texts,  device=args.device)

    # FAISS indices
    print("[INFO] Building FAISS indices...")
    file_index = build_faiss_index(file_vecs)
    sym_index = build_faiss_index(sym_vecs)

    # Save indices + ids
    file_ids = file_df["uid"].tolist()
    symbol_ids = sym_df["uid"].tolist()
    file_index_path, file_ids_path = save_index(file_index, file_ids, out_dir, "file")
    symbol_index_path, symbol_ids_path = save_index(sym_index, symbol_ids, out_dir, "symbol")

    # Save meta
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "units_path": df_path}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved to: {out_dir}")
    print(f" - units: {df_path}")
    print(f" - file.index / file_ids.json")
    print(f" - symbol.index / symbol_ids.json")
    print(f" - meta: {meta_path}")

    # Optional query
    if args.query:
        print(f"\n[QUERY] {args.query}")
        hits = hybrid_query(args.query, out_dir, topk=args.topk)
        # Join with dataframe for pretty print
        df_idx = df.set_index("uid")
        for h in hits:
            row = df_idx.loc[h["uid"]]
            print(f"[{h['source']}] score={h['score']:.3f} | {row['file_path']} | "
                  f"{row.get('symbol_type') or 'file'}::{row.get('symbol_name') or ''}")
            # brief preview
            if row.get("summary"):
                print("  summary:", (row["summary"][:220] + "...") if len(row["summary"]) > 220 else row["summary"])
            else:
                code = row.get("code") or ""
                print("  code:", (code[:220] + "...") if len(code) > 220 else code)

if __name__ == "__main__":
    main()
