#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

# Optional: PyVis (for interactive graph)
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except Exception:
    PYVIS_AVAILABLE = False

from dotenv import load_dotenv

# ===================== Load environment variables =====================

load_dotenv()

# ===================== Configuration =====================
DEFAULT_V1_DIR = os.path.join(os.getcwd(), ".cache", "graphs")                # verbose v1 (nodes+edges)
DEFAULT_V2_DIR = os.path.join(os.getcwd(), ".cache", "graphs_compact")        # compact v2 (single)
DEFAULT_MANIFEST_BASE = os.path.join(os.getcwd(), ".cache", "graphs_compact") # manifests live under <v2>/<graph_id>/

os.makedirs(DEFAULT_V1_DIR, exist_ok=True)
os.makedirs(DEFAULT_V2_DIR, exist_ok=True)

SKIP_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "build", "dist", "node_modules",
    ".mypy_cache", ".pytest_cache", "target", "out", ".gradle", ".idea"
}
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2MB per file safety limit

# Language sets by extension
PY_EXTS   = {"py"}
JS_TS_EXTS= {"js", "jsx", "ts", "tsx", "mjs", "cjs"}
JAVA_EXTS = {"java"}
GO_EXTS   = {"go"}
C_EXTS    = {"c", "h"}
CPP_EXTS  = {"cc", "cpp", "cxx", "hpp", "hh", "hxx", "h++", "c++"}
RUST_EXTS = {"rs"}
RUBY_EXTS = {"rb"}
PHP_EXTS  = {"php"}
KT_EXTS   = {"kt", "kts"}

# ===================== Utilities =====================

def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sanitize_repo_name(name: str) -> str:
    return re.sub(r"[^\w\-\.]+", "-", name).strip("-")

def parse_github_url(url: str):
    """
    Supports:
      - https://github.com/<owner>/<repo>
      - https://github.com/<owner>/<repo>.git
      - https://github.com/<owner>/<repo>/tree/<branch>/<optional subpath>
      - https://github.com/<owner>/<repo>/blob/<branch>/<file>
    Returns: dict(owner, repo, branch=None, subpath="", kind)
    """
    url = url.strip()
    url = re.sub(r"\.git$", "", url)
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)(?:/(tree|blob)/([^/]+)(/.*)?)?$", url)
    if not m:
        raise ValueError("Unrecognized GitHub URL. Expect https://github.com/<owner>/<repo>[...]")
    owner, repo, kind, branch, subpath = m.groups()
    repo = sanitize_repo_name(repo)
    subpath = (subpath or "").lstrip("/")
    return {"owner": owner, "repo": repo, "branch": branch, "subpath": subpath, "kind": kind or "repo"}

def http_get_json(url: str, token: Optional[str] = None):
    headers = {"User-Agent": "streamlit-kg-app"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        return json.load(io.TextIOWrapper(resp, encoding="utf-8"))

def get_default_branch(owner: str, repo: str, token: Optional[str] = None) -> str:
    try:
        info = http_get_json(f"https://api.github.com/repos/{owner}/{repo}", token=token)
        if info.get("default_branch"):
            return info["default_branch"]
    except Exception:
        pass
    for candidate in ("main", "master"):
        if can_download_zip(owner, repo, candidate):
            return candidate
    return "main"

def can_download_zip(owner: str, repo: str, branch: str) -> bool:
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    try:
        req = Request(url, headers={"User-Agent": "streamlit-kg-app"})
        with urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False

def download_repo_zip(owner: str, repo: str, branch: str) -> bytes:
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    req = Request(url, headers={"User-Agent": "streamlit-kg-app"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()

def unzip_to_temp(zip_bytes: bytes) -> str:
    tmpdir = tempfile.mkdtemp(prefix="ghrepo_")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmpdir)
    entries = [os.path.join(tmpdir, d) for d in os.listdir(tmpdir)]
    dirs = [p for p in entries if os.path.isdir(p)]
    if not dirs:
        raise RuntimeError("Unexpected ZIP structure.")
    # pick largest
    return max(dirs, key=lambda p: sum(len(files) for _, _, files in os.walk(p)))

def detect_lang_by_ext(ext: str) -> str:
    e = ext.lower().lstrip(".")
    if e in PY_EXTS: return "python"
    if e in JS_TS_EXTS: return "javascript"
    if e in JAVA_EXTS: return "java"
    if e in GO_EXTS: return "go"
    if e in CPP_EXTS or e in C_EXTS: return "cpp" if e in CPP_EXTS else "c"
    if e in RUST_EXTS: return "rust"
    if e in RUBY_EXTS: return "ruby"
    if e in PHP_EXTS: return "php"
    if e in KT_EXTS: return "kotlin"
    return "text"

def iter_repo_files(root_dir: str, subpath: Optional[str] = None, max_files: Optional[int] = None):
    base = os.path.join(root_dir, subpath) if subpath else root_dir
    count = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            abs_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(abs_path, root_dir).replace("\\", "/")
            yield abs_path, rel_path
            count += 1
            if max_files and count >= max_files:
                return
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

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def fingerprint(owner: str, repo: str, branch: str, subpath: Optional[str], url: str) -> str:
    payload = f"{owner}/{repo}@{branch}:{subpath or ''}|{url}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

def load_json_autoz(path: str) -> Dict[str, Any]:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

# ---------- Wiki output path helpers ----------

from typing import Dict, Any

def wiki_default_output_path(graph_meta: Dict[str, Any]) -> str:
    """
    Save XML like we save graphs:
    .cache/wiki_structures/<owner>__<repo>__<branch>__<graph_id>__wiki.xml
    """
    graph_id = graph_meta.get("graph_id", "graph")
    out_dir = ensure_dir(os.path.join(os.getcwd(), ".cache", "wiki_structures"))
    fname = f"{graph_meta.get('owner','owner')}__{graph_meta.get('repo','repo')}__{graph_meta.get('branch','branch')}__{graph_id}__wiki.xml"
    fname = fname.replace("/", "_")
    return os.path.join(out_dir, fname)

def wiki_partial_output_path(graph_meta: Dict[str, Any], shard_name: str) -> str:
    """
    Consistent naming for shard partials:
    .cache/wiki_structures/<graph_id>/partials/<owner>__<repo>__<branch>__<graph_id>__partial__<shard>.xml
    """
    graph_id = graph_meta.get("graph_id", "graph")
    base_dir = ensure_dir(os.path.join(os.getcwd(), ".cache", "wiki_structures", graph_id, "partials"))
    fname = f"{graph_meta.get('owner','owner')}__{graph_meta.get('repo','repo')}__{graph_meta.get('branch','branch')}__{graph_id}__partial__{shard_name}.xml"
    fname = fname.replace("/", "_")
    return os.path.join(base_dir, fname)


# ===================== Language Parsers =====================

def parse_python_text(py_src: str):
    classes, imports, functions = [], [], []
    try:
        tree = ast.parse(py_src)
    except SyntaxError:
        return classes, imports, functions
    if hasattr(tree, "body"):
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) or isinstance(n, ast.AsyncFunctionDef):
                functions.append(n.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"type": "import", "module": alias.name, "names": [], "level": 0})
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [alias.name for alias in node.names]
            imports.append({"type": "from", "module": mod, "names": names, "level": node.level or 0})
    return classes, imports, functions

_ident = r"[A-Za-z_][A-Za-z0-9_]*"
_js_ident = r"[A-Za-z_$][A-Za-z0-9_$]*"

def parse_js_ts_text(src: str):
    classes, imports, functions = [], [], []
    for m in re.finditer(rf"\bclass\s+({_js_ident})\b", src):
        classes.append(m.group(1))
    for m in re.finditer(r"""import\s+(?:[\s\S]*?\s+from\s+)?'"['"]""", src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    for m in re.finditer(r"""require\(\s*['"]([^'"\)""", src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    for m in re.finditer(rf"\bfunction\s+({_js_ident})\s*\(", src):
        functions.append(m.group(1))
    for m in re.finditer(rf"\bexport\s+function\s+({_js_ident})\s*\(", src):
        functions.append(m.group(1))
    for m in re.finditer(rf"\b(?:const|let|var)\s+({_js_ident})\s*=\s*function\b", src):
        functions.append(m.group(1))
    for m in re.finditer(rf"\b(?:const|let|var)\s+({_js_ident})\s*=\s*\(", src):
        functions.append(m.group(1))
    return classes, imports, list(dict.fromkeys(functions))

def parse_java_text(src: str):
    classes, imports, functions = [], [], []
    for m in re.finditer(rf"\b(class|interface|enum)\s+({_ident})\b", src):
        classes.append(m.group(2))
    for m in re.finditer(r"\bimport\s+([a-zA-Z0-9_\.]+)\s*;", src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    return classes, imports, functions

def parse_go_text(src: str):
    classes, imports, functions = [], [], []
    for m in re.finditer(r'import\s+"([^"]+)"', src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    block = re.search(r"import\s*\((.*?)\)", src, re.S)
    if block:
        for m in re.finditer(r'"([^"]+)"', block.group(1)):
            imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    for m in re.finditer(rf"\btype\s+({_ident})\s+struct\b", src):
        classes.append(m.group(1))
    for m in re.finditer(rf"\bfunc\s+(?:\([^)]+\)\s*)?({_ident})\s*\(", src):
        functions.append(m.group(1))
    return classes, imports, functions

def parse_c_cpp_text(src: str):
    """
    Heuristic parsing for C/C++:
      - #include "..." and #include <...>  -> imports
      - class/struct <Name>                -> classes
      - top-level function definitions     -> functions (declarations excluded)
    """
    import re

    classes, imports, functions = [], [], []

    # Includes: #include <header> or #include "header"
    include_pattern = r'#\s*include\s*<"[>"]'
    for m in re.finditer(include_pattern, src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})

    # Identifier pattern
    _ident = r"[A-Za-z_][A-Za-z0-9_]*"

    # class/struct names
    for m in re.finditer(rf"\b(class|struct)\s+({_ident})\b", src):
        classes.append(m.group(2))

    # Heuristic top-level function definition (requires a body '{', avoids ';' declarations)
    # Examples matched:
    #   int foo(int a) { ... }
    #   static inline MyType ns::Class::method(T x) { ... }
    #   template<typename T> T bar(T x) { ... }
    func_pattern = (
        r"(?m)^[ \t]*"                 # line start
        r"[A-Za-z_][\w:\s\*\&\<\>]*\s+" # return type / qualifiers
        r"(" + _ident + r")"           # function name (capture)
        r"\s*\([^;]*\)"                # args (not containing ';')
        r"\s*\{"                       # opening brace of body
    )
    for m in re.finditer(func_pattern, src):
        functions.append(m.group(1))

    # De-dup preserve order
    functions = list(dict.fromkeys(functions))
    return classes, imports, functions

def parse_rust_text(src: str):
    classes, imports, functions = [], [], []
    for m in re.finditer(r"\buse\s+([A-Za-z0-9_:\{\}\*,\s]+);", src):
        mod = re.sub(r"\s+", " ", m.group(1)).strip()
        imports.append({"type": "import", "module": mod, "names": [], "level": 0})
    for m in re.finditer(rf"\b(struct|enum)\s+({_ident})\b", src):
        classes.append(m.group(2))
    for m in re.finditer(rf"\bfn\s+({_ident})\s*\(", src):
        functions.append(m.group(1))
    return classes, imports, functions

def parse_ruby_text(src: str):
    classes, imports, functions = [], [], []
    for m in re.finditer(rf"\bclass\s+({_ident})\b", src):
        classes.append(m.group(1))
    for m in re.finditer(r"""(?:require|require_relative)\s+'"['"]""", src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    for m in re.finditer(rf"(?m)^\s*def\s+({_ident})\b", src):
        functions.append(m.group(1))
    return classes, imports, functions

def parse_php_text(src: str):
    classes, imports, functions = [], [], []
    for m in re.finditer(rf"\bclass\s+({_ident})\b", src):
        classes.append(m.group(1))
    for m in re.finditer(rf"\bfunction\s+({_ident})\s*\(", src):
        functions.append(m.group(1))
    for m in re.finditer(rf"\buse\s+([A-Za-z0-9_\\]+)\s*;", src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    return classes, imports, functions

def parse_kotlin_text(src: str):
    classes, imports, functions = [], [], []
    for m in re.finditer(rf"\b(class|object|interface)\s+({_ident})\b", src):
        classes.append(m.group(2))
    for m in re.finditer(rf"\bimport\s+([A-Za-z0-9_\.]+)\s*", src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    for m in re.finditer(rf"\bfun\s+({_ident})\s*\(", src):
        functions.append(m.group(1))
    return classes, imports, functions

def parse_any_text_by_ext(ext: str, text: str):
    e = ext.lower().lstrip(".")
    if e in PY_EXTS: return parse_python_text(text)
    if e in JS_TS_EXTS: return parse_js_ts_text(text)
    if e in JAVA_EXTS: return parse_java_text(text)
    if e in GO_EXTS: return parse_go_text(text)
    if e in C_EXTS or e in CPP_EXTS: return parse_c_cpp_text(text)
    if e in RUST_EXTS: return parse_rust_text(text)
    if e in RUBY_EXTS: return parse_ruby_text(text)
    if e in PHP_EXTS: return parse_php_text(text)
    if e in KT_EXTS: return parse_kotlin_text(text)
    return [], [], []



# ===================== Relevance Guard: Signals, Allowed Sections, Pruning =====================

def _norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = s.replace("&", "and")
    return s

def _derive_repo_signals_from_paths(file_paths: List[str]) -> dict:
    """
    Derive boolean signals from file paths to gate allowed sections.
    """
    exts = set(os.path.splitext(p)[1].lower() for p in file_paths)
    has_py = any(p.endswith(".py") for p in file_paths)
    has_js_ts = any(p.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")) for p in file_paths)
    has_java = any(p.endswith(".java") for p in file_paths)
    has_cpp = any(p.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx")) for p in file_paths)
    has_go = any(p.endswith(".go") for p in file_paths)
    has_rust = any(p.endswith(".rs") for p in file_paths)
    has_kotlin = any(p.endswith((".kt", ".kts")) for p in file_paths)
    has_php = any(p.endswith(".php") for p in file_paths)
    has_ruby = any(p.endswith(".rb") for p in file_paths)

    has_examples_dir = any("/examples/" in p.lower() or p.lower().startswith("examples/") for p in file_paths)
    has_notebooks = any(p.endswith(".ipynb") for p in file_paths) or any("/notebook" in p.lower() or "/notebooks" in p.lower() for p in file_paths)
    has_docs = any(p.lower() in ("readme", "readme.md") or p.lower().startswith("docs/") or "/docs/" in p.lower() or p.lower().endswith(("mkdocs.yml", "mkdocs.yaml")) for p in file_paths)
    has_scripts = any("/scripts/" in p.lower() or p.lower().startswith("scripts/") for p in file_paths)
    has_bench = any("bench" in p.lower() or "perf" in p.lower() for p in file_paths)
    has_infra = any(any(tok in p.lower() for tok in ["dockerfile", "docker-compose", "helm", "k8s", "terraform", "ansible", ".github/workflows", "cloudbuild", "jenkinsfile"]) for p in file_paths)
    has_frontend = any(any(tok in p.lower() for tok in ["package.json", "vite.config", "webpack.config", "src/components", "src/app", "next.config"]) for p in file_paths)
    has_backend = any(any(tok in p.lower() for tok in ["api/", "/controllers/", "/services/", "/routes/"]) for p in file_paths)
    has_data = any(any(tok in p.lower() for tok in ["data/", "/dataset", "/etl", "/pipeline", "/db/", ".sql", "migrations/"]) for p in file_paths)
    has_models = any(any(tok in p.lower() for tok in ["model", "ml", "nn", "inference", "training", "weights"]) for p in file_paths)

    return {
        "has_py": has_py, "has_js_ts": has_js_ts, "has_java": has_java, "has_cpp": has_cpp,
        "has_go": has_go, "has_rust": has_rust, "has_kotlin": has_kotlin, "has_php": has_php, "has_ruby": has_ruby,
        "has_examples_dir": has_examples_dir, "has_notebooks": has_notebooks, "has_docs": has_docs,
        "has_scripts": has_scripts, "has_bench": has_bench, "has_infra": has_infra,
        "has_frontend": has_frontend, "has_backend": has_backend, "has_data": has_data, "has_models": has_models,
    }

def _allowed_forbidden_sections(signals: dict) -> tuple[list, list]:
    """
    Decide allowed vs. forbidden section titles from signals.
    Titles are case-insensitive; use normalized comparisons.
    """
    # Canonical section set we aim for
    base_allowed = [
        "overview",
        "system architecture",
        "core features",
        "data management/flow",
        "deployment/infrastructure",
        "extensibility and customization",
    ]

    # Conditionally add
    if signals["has_frontend"]:
        base_allowed.append("frontend components")
    if signals["has_backend"] or signals["has_py"] or signals["has_java"] or signals["has_go"] or signals["has_rust"] or signals["has_php"] or signals["has_kotlin"]:
        base_allowed.append("backend systems")
    if signals["has_models"]:
        base_allowed.append("model integration")
    if signals["has_examples_dir"] or signals["has_notebooks"]:
        base_allowed.append("examples and notebooks")

    # Explicitly forbidden if no evidence
    forbidden = []
    if not (signals["has_examples_dir"] or signals["has_notebooks"]):
        forbidden.extend(["examples", "notebooks", "examples and notebooks"])
    if not signals["has_frontend"]:
        forbidden.append("frontend components")
    if not (signals["has_backend"] or signals["has_py"] or signals["has_java"] or signals["has_go"] or signals["has_rust"] or signals["has_php"] or signals["has_kotlin"]):
        forbidden.append("backend systems")
    if not signals["has_models"]:
        forbidden.append("model integration")

    # Some noisy patterns we generally do not want as sections
    forbidden.extend([
        "text examples",
        "c++ examples",
        "python examples",
        "random scripts",
        "playground",
        "misc",
    ])

    # normalize
    allowed_norm = sorted(set(_norm_title(x) for x in base_allowed))
    forbidden_norm = sorted(set(_norm_title(x) for x in forbidden))
    return allowed_norm, forbidden_norm

def _collect_all_paths_from_v1like(graph_v1: dict) -> list[str]:
    return [n.get("path", n.get("label", "")) for n in graph_v1.get("nodes", []) if n.get("type") == "file"]

def _collect_all_paths_from_compact(compact: dict) -> list[str]:
    return [f.get("path", "") for f in compact.get("files", [])]

def _prune_and_renumber_wiki(xml_text: str,
                             allowed_sections_norm: list[str],
                             file_paths_set: set[str]) -> str:
    """
    Parse wiki_structure XML:
      - remove sections whose title is not in allowed set
      - drop pages with no relevant_files or whose files don't exist
      - remove empty sections
      - renumber section/page IDs sequentially
    Return cleaned XML as string.
    """
    import xml.etree.ElementTree as ET

    def _exists_file(fp: str) -> bool:
        return fp in file_paths_set

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        # Try wrapping or return as-is (it will be refined later)
        return xml_text

    if root.tag != "wiki_structure":
        return xml_text

    # Build lists
    sections_el = root.find("sections")
    pages_el = root.find("pages")
    if sections_el is None or pages_el is None:
        return xml_text

    # Normalize section titles and mark allowed ones
    keep_sections = []
    for s in list(sections_el.findall("section")):
        title = (s.findtext("title") or "").strip()
        if not title:
            continue
        if _norm_title(title) in allowed_sections_norm:
            keep_sections.append(s)

    # Replace sections with filtered
    for s in list(sections_el):
        sections_el.remove(s)
    for s in keep_sections:
        sections_el.append(s)

    # Check pages: keep only those with at least one valid file path
    page_elems = list(pages_el.findall("page"))
    keep_pages = []
    for p in page_elems:
        rf = p.find("relevant_files")
        valid_files = []
        if rf is not None:
            for fp in rf.findall("file_path"):
                path_txt = (fp.text or "").strip()
                if path_txt and _exists_file(path_txt):
                    valid_files.append(path_txt)
        if valid_files:
            # Replace with only valid files
            # Clear existing and add valid
            for child in list(rf):
                rf.remove(child)
            for v in sorted(set(valid_files)):
                ET.SubElement(rf, "file_path").text = v
            keep_pages.append(p)
        # else: drop page

    # Reset pages list
    for p in page_elems:
        pages_el.remove(p)
    for p in keep_pages:
        pages_el.append(p)

    # Remove page_refs to dropped pages and drop empty sections
    valid_page_ids = set(p.get("id") for p in pages_el.findall("page"))
    kept_sections_final = []
    for s in list(sections_el.findall("section")):
        pages_container = s.find("pages")
        # Rebuild page_ref list
        kept_refs = []
        if pages_container is not None:
            for r in list(pages_container.findall("page_ref")):
                tid = (r.text or "").strip()
                if tid in valid_page_ids:
                    kept_refs.append(tid)
        # Replace refs
        if pages_container is None:
            pages_container = ET.SubElement(s, "pages")
        for r in list(pages_container):
            pages_container.remove(r)
        for tid in kept_refs:
            ET.SubElement(pages_container, "page_ref").text = tid
        if kept_refs:
            kept_sections_final.append(s)
        # else drop section

    # Reset sections list
    for s in list(sections_el):
        sections_el.remove(s)
    for s in kept_sections_final:
        sections_el.append(s)

    # Renumber IDs sequentially
    # Build mapping section-old -> section-1..N by order
    new_sec_ids = {}
    for i, s in enumerate(list(sections_el.findall("section")), start=1):
        new_id = f"section-{i}"
        new_sec_ids[s.get("id")] = new_id
        s.set("id", new_id)

    # Renumber pages and update page_refs + parent_section
    page_old_to_new = {}
    for i, p in enumerate(list(pages_el.findall("page")), start=1):
        new_id = f"page-{i}"
        page_old_to_new[p.get("id")] = new_id
        p.set("id", new_id)
        # parent_section may refer to old ids; fix if possible
        ps = p.find("parent_section")
        if ps is not None:
            old = (ps.text or "").strip()
            if old in new_sec_ids:
                ps.text = new_sec_ids[old]
            # else keep as-is (or set to section-1 if none)
            if not ps.text:
                ps.text = "section-1"

    # Rewrite page_ref texts
    for s in list(sections_el.findall("section")):
        pages_container = s.find("pages")
        if pages_container is None:
            pages_container = ET.SubElement(s, "pages")
        for r in list(pages_container.findall("page_ref")):
            tid = (r.text or "").strip()
            if tid in page_old_to_new:
                r.text = page_old_to_new[tid]

    # Serialize back
    return ET.tostring(root, encoding="unicode")


# ---------- Cache helpers ----------

def wiki_default_output_path(graph_meta: Dict[str, Any]) -> str:
    """
    Save XML like we save graphs:
    .cache/wiki_structures/<owner>__<repo>__<branch>__<graph_id>__wiki.xml
    """
    graph_id = graph_meta.get("graph_id", "graph")
    out_dir = ensure_dir(os.path.join(os.getcwd(), ".cache", "wiki_structures"))
    fname = f"{graph_meta.get('owner','owner')}__{graph_meta.get('repo','repo')}__{graph_meta.get('branch','branch')}__{graph_id}__wiki.xml"
    fname = fname.replace("/", "_")
    return os.path.join(out_dir, fname)

def wiki_partial_output_path(graph_meta: Dict[str, Any], shard_name: str) -> str:
    """
    Consistent naming for shard partials:
    .cache/wiki_structures/<graph_id>/partials/<owner>__<repo>__<branch>__<graph_id>__partial__<shard>.xml
    """
    graph_id = graph_meta.get("graph_id", "graph")
    base_dir = ensure_dir(os.path.join(os.getcwd(), ".cache", "wiki_structures", graph_id, "partials"))
    fname = f"{graph_meta.get('owner','owner')}__{graph_meta.get('repo','repo')}__{graph_meta.get('branch','branch')}__{graph_id}__partial__{shard_name}.xml"
    fname = fname.replace("/", "_")
    return os.path.join(base_dir, fname)

def expected_verbose_v1_path(meta: Dict[str, Any], v1_dir: str) -> str:
    name = f"{meta.get('owner','owner')}__{meta.get('repo','repo')}__{meta.get('branch','branch')}__{meta.get('graph_id','graph')}.json"
    return os.path.join(v1_dir, name.replace("/", "_"))

def expected_compact_v2_paths(meta: Dict[str, Any], v2_dir: str) -> Dict[str, str]:
    """
    Return candidate paths for compact v2 single file (gz/plain) and sharded manifest (gz/plain).
    """
    graph_id = meta.get("graph_id", "graph")
    base_single = f"{meta.get('owner','owner')}__{meta.get('repo','repo')}__{meta.get('branch','branch')}__{graph_id}__v2.json"
    single_gz = os.path.join(v2_dir, base_single.replace("/", "_") + ".gz")
    single_json = os.path.join(v2_dir, base_single.replace("/", "_"))
    man_dir = os.path.join(v2_dir, graph_id)
    man_json = os.path.join(man_dir, "manifest.json")
    man_gz = man_json + ".gz"
    return {
        "single_gz": single_gz,
        "single_json": single_json,
        "manifest_json": man_json,
        "manifest_gz": man_gz,
        "manifest_dir": man_dir,
    }

def first_existing_path(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


# ===================== Graph Builders =====================

def build_repo_kg_v1(root_dir: str, subpath: Optional[str] = None, progress=None, max_files: Optional[int] = None):
    """
    Verbose v1: nodes + edges
    """
    graph = {"nodes": [], "edges": []}
    nodes_index = {}
    totals = {"files": 0, "classes": 0, "imports": 0, "functions": 0}

    def add_node(node_id, node_type, label, **props):
        if node_id in nodes_index: return
        node = {"id": node_id, "type": node_type, "label": label}
        node.update(props)
        graph["nodes"].append(node)
        nodes_index[node_id] = node

    def add_edge(source, target, edge_type, **props):
        edge = {"source": source, "target": target, "type": edge_type}
        edge.update(props)
        graph["edges"].append(edge)

    file_list = list(iter_repo_files(root_dir, subpath, max_files=max_files))
    n = len(file_list)

    for i, (abs_path, rel_path) in enumerate(file_list, start=1):
        if progress:
            progress.progress(min(i / max(n, 1), 1.0), text=f"Scanning {rel_path} ({i}/{n})")

        ext = os.path.splitext(rel_path)[1]
        lang = detect_lang_by_ext(ext)
        file_text = read_text_file(abs_path)

        file_node_id = f"file:{rel_path}"
        add_node(file_node_id, "file", rel_path, path=rel_path, lang=lang)
        totals["files"] += 1

        if not file_text:
            continue

        classes, imports, functions = parse_any_text_by_ext(ext, file_text)

        for cls in classes:
            class_node_id = f"class:{rel_path}#{cls}"
            add_node(class_node_id, "class", cls, file=rel_path, lang=lang)
            add_edge(file_node_id, class_node_id, "FILE_CONTAINS_CLASS")
        totals["classes"] += len(classes)

        for fn in functions:
            func_node_id = f"function:{rel_path}#{fn}"
            add_node(func_node_id, "function", fn, file=rel_path, lang=lang)
            add_edge(file_node_id, func_node_id, "FILE_CONTAINS_FUNCTION")
        totals["functions"] += len(functions)

        for imp in imports:
            module = imp.get("module") or ""
            if not module: continue
            import_node_id = f"import:{module}"
            add_node(import_node_id, "import", module, kind=imp.get("type"))
            add_edge(file_node_id, import_node_id, "FILE_IMPORTS", detail=imp.get("type"))
        totals["imports"] += len(imports)

    return graph, totals

def save_graph_v1(cache_dir: str, meta: dict, graph: dict) -> str:
    ensure_dir(cache_dir)
    filename = f"{meta['owner']}__{meta['repo']}__{meta['branch']}__{meta['graph_id']}.json"
    filename = filename.replace("/", "_")
    out_path = os.path.join(cache_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **graph}, f, ensure_ascii=False, indent=2)
    return out_path

def build_repo_compact_v2(root_dir: str, subpath: Optional[str] = None, progress=None, max_files: Optional[int] = None):
    """
    Compact v2:
      - Global dicts.imports (deduped)
      - Per-file records: path, lang, classes[], functions[], imports[] (indices)
    """
    files = []
    import_to_idx = {}
    imports_list = []

    def get_import_index(name: str) -> int:
        if name not in import_to_idx:
            import_to_idx[name] = len(imports_list)
            imports_list.append(name)
        return import_to_idx[name]

    file_list = list(iter_repo_files(root_dir, subpath, max_files=max_files))
    n = len(file_list)
    totals = {"files": 0, "classes": 0, "functions": 0, "imports": 0}

    for i, (abs_path, rel_path) in enumerate(file_list, start=1):
        if progress:
            progress.progress(min(i / max(n, 1), 1.0), text=f"Scanning {rel_path} ({i}/{n})")

        ext = os.path.splitext(rel_path)[1]
        lang = detect_lang_by_ext(ext)
        text = read_text_file(abs_path)

        rec = {"path": rel_path, "lang": lang, "classes": [], "functions": [], "imports": []}
        totals["files"] += 1

        if not text:
            files.append(rec)
            continue

        classes, imports, functions = parse_any_text_by_ext(ext, text)

        if classes:
            rec["classes"] = sorted(set(classes))
            totals["classes"] += len(rec["classes"])

        if functions:
            rec["functions"] = sorted(set(functions))
            totals["functions"] += len(rec["functions"])

        if imports:
            mods = []
            for imp in imports:
                module = (imp.get("module") or "").strip()
                if not module:
                    continue
                mods.append(module)
            if mods:
                mods = sorted(set(mods))
                rec["imports"] = [get_import_index(m) for m in mods]
                totals["imports"] += len(mods)

        files.append(rec)

    compact = {
        "meta": {},  # to be filled by caller
        "dicts": {"imports": imports_list},
        "files": files
    }
    return compact, totals

def save_json_gz(obj: dict, path: str):
    ensure_dir(os.path.dirname(path))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

def save_compact_graph_v2(cache_dir: str, meta: dict, compact: dict, gzip_out: bool = True) -> str:
    compact["meta"] = meta | {"schema": "v2-compact"}
    base_name = f"{meta['owner']}__{meta['repo']}__{meta['branch']}__{meta['graph_id']}__v2.json"
    base_name = base_name.replace("/", "_")
    out_path = os.path.join(cache_dir, base_name + (".gz" if gzip_out else ""))
    if gzip_out:
        save_json_gz(compact, out_path)
    else:
        ensure_dir(os.path.dirname(out_path))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, separators=(",", ":"))
    return out_path

def shard_compact_by_top_dir(compact: dict, out_dir: str, gzip_out: bool = True) -> dict:
    """
    Writes shards and a manifest:
      - out_dir/manifest.json[.gz]
      - out_dir/shards/shard__<topdir>.json[.gz]
    """
    imports = compact.get("dicts", {}).get("imports", [])
    files = compact.get("files", [])
    groups = defaultdict(list)
    for f in files:
        top = (f["path"].split("/", 1)[0]) if "/" in f["path"] else ""
        groups[top].append(f)

    ensure_dir(out_dir)
    shards_dir = os.path.join(out_dir, "shards")
    ensure_dir(shards_dir)

    meta = compact.get("meta", {})
    shard_records = []
    for top, flist in groups.items():
        shard = {"meta": meta | {"shard": top}, "dicts": {"imports": imports}, "files": flist}
        name = f"shard__{top or '_root'}.json" + (".gz" if gzip_out else "")
        path = os.path.join(shards_dir, name)
        if gzip_out:
            save_json_gz(shard, path)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(shard, f, ensure_ascii=False, separators=(",", ":"))
        shard_records.append({"topdir": top, "path": os.path.relpath(path, out_dir)})

    manifest = {"meta": meta, "shards": shard_records, "imports_count": len(imports), "files_total": len(files)}
    man_path = os.path.join(out_dir, "manifest.json" + (".gz" if gzip_out else ""))
    if gzip_out:
        save_json_gz(manifest, man_path)
    else:
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, separators=(",", ":"))
    return manifest

def expand_compact_to_v1(compact: dict) -> dict:
    """
    Reconstruct a v1-like graph (nodes+edges) from compact v2 for viewer/preview/wiki.
    """
    nodes, edges = [], []
    imports = compact.get("dicts", {}).get("imports", [])
    files = compact.get("files", [])

    import_nodes_added = set()

    for f in files:
        file_id = f"file:{f['path']}"
        nodes.append({"id": file_id, "type": "file", "label": f["path"], "path": f["path"], "lang": f.get("lang", "")})

        for cname in f.get("classes", []):
            cid = f"class:{f['path']}#{cname}"
            nodes.append({"id": cid, "type": "class", "label": cname, "file": f["path"], "lang": f.get("lang", "")})
            edges.append({"source": file_id, "target": cid, "type": "FILE_CONTAINS_CLASS"})

        for gname in f.get("functions", []):
            gid = f"function:{f['path']}#{gname}"
            nodes.append({"id": gid, "type": "function", "label": gname, "file": f["path"], "lang": f.get("lang", "")})
            edges.append({"source": file_id, "target": gid, "type": "FILE_CONTAINS_FUNCTION"})

        for idx in f.get("imports", []):
            if idx < 0 or idx >= len(imports): continue
            mod = imports[idx]
            iid = f"import:{mod}"
            if iid not in import_nodes_added:
                nodes.append({"id": iid, "type": "import", "label": mod, "kind": "import"})
                import_nodes_added.add(iid)
            edges.append({"source": file_id, "target": iid, "type": "FILE_IMPORTS"})

    return {"meta": compact.get("meta", {}), "nodes": nodes, "edges": edges}


# ===================== LLM (Local) for Wiki =====================

def qgenie_generate_xml(
    prompt_text: str,
) -> str:
    from qgenie import ChatMessage, QGenieClient

    client = QGenieClient()
 
 
    chat_response = client.chat(
    messages=[
        ChatMessage(role="user", content=prompt_text)
    ],
    # model = "qwen2.5-14b-1m",
    # max_seconds=60,
)
 
    return chat_response.first_content

def validate_or_wrap_xml(text: str, root_tag: str) -> str:
    t = text.strip()
    start = t.find(f"<{root_tag}")
    end = t.rfind(f"</{root_tag}>")
    if start != -1 and end != -1:
        return t[start:end + len(f"</{root_tag}>")].strip()
    return f"<{root_tag}>\n{text}\n</{root_tag}>"

# ===================== Streamlit UI =====================

st.set_page_config(page_title="GitHub → Knowledge Graph (Sharded v2 + Wiki)", layout="wide")
st.title("📚 GitHub → Knowledge Graph (Polyglot) → 🧩 Compact v2 + Shards → 📄 Wiki")

with st.sidebar:
    st.header("Input")
    gh_url = st.text_input(
        "GitHub URL",
        help="Examples:\n- https://github.com/pallets/flask\n- https://github.com/pallets/flask/tree/main/src\n- https://github.com/owner/repo/blob/main/path/to/file.js"
    )
    token = st.text_input("GitHub token (optional)", type="password",
                          help="Improves API rate limits. Create a fine-grained token with minimal permissions.")
    max_files = st.number_input("Max files to scan (0 = no limit)", min_value=0, value=0, step=25)

    st.divider()
    st.header("Storage")
    storage_format = st.radio(
        "Build format",
        options=["Compact v2 (+ optional shards)", "Verbose v1 (legacy)"],
        index=0,
        help="Compact v2 is much smaller; shard by top-level directory for huge repos."
    )
    gzip_out = st.checkbox("Gzip output (.json.gz)", value=True)
    do_shard = st.checkbox("Create shards (per top-level dir)", value=True, help="Writes manifest + shard files for v2.")

    st.divider()
    st.header("Cache directories")
    v1_dir = st.text_input("Verbose v1 directory", value=DEFAULT_V1_DIR)
    v2_dir = st.text_input("Compact v2 directory", value=DEFAULT_V2_DIR)

    run_btn = st.button("Build Knowledge Graph", type="primary", use_container_width=True)

tab_build, tab_preview,  tab_wiki = st.tabs([
    "🔨 Build",
    "👀 Preview / Explore",
    "📄 Generate Wiki"
])

# ===================== Build tab =====================
if run_btn:
    with tab_build:
        st.subheader("Build Log")
        try:
            if not gh_url:
                st.error("Please enter a GitHub URL.")
                st.stop()

            parts = parse_github_url(gh_url)
            owner = parts["owner"]
            repo = parts["repo"]
            branch = parts["branch"] or get_default_branch(owner, repo, token=token)
            subpath = parts["subpath"] or None

            st.write(f"**Repo:** `{owner}/{repo}`  |  **Branch:** `{branch}`  |  **Subpath:** `{subpath or '.'}`")

            with st.status("Downloading repository...", expanded=False) as status:
                zip_bytes = download_repo_zip(owner, repo, branch)
                status.update(label="Downloaded ZIP ✔️")

            with st.status("Unzipping archive...", expanded=False) as status:
                repo_root = unzip_to_temp(zip_bytes)
                status.update(label="Unzipped ✔️")

            st.write("Scanning repository files (polyglot)...")
            progress = st.progress(0.0, text="Starting...")
            limit = max_files if max_files and max_files > 0 else None

            g_id = fingerprint(owner, repo, branch, subpath, gh_url)
            meta_common = {
                "source_url": gh_url,
                "owner": owner,
                "repo": repo,
                "branch": branch,
                "subpath": subpath or "",
                "created_at": now_iso(),
                "graph_id": g_id,
            }

            if storage_format == "Verbose v1 (legacy)":
                graph_v1, totals = build_repo_kg_v1(repo_root, subpath=subpath, progress=progress, max_files=limit)
                meta = meta_common | {"totals": totals}
                out_path = save_graph_v1(v1_dir, meta, graph_v1)
                st.success("Verbose graph built and saved!")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Files", totals["files"]); c2.metric("Classes", totals["classes"])
                c3.metric("Functions", totals["functions"]); c4.metric("Imports", totals["imports"])
                st.code(out_path, language="bash")
                with open(out_path, "r", encoding="utf-8") as f:
                    st.download_button("Download Graph JSON (v1)", f.read(), file_name=os.path.basename(out_path),
                                       mime="application/json", use_container_width=True)
            else:
                compact, totals = build_repo_compact_v2(repo_root, subpath=subpath, progress=progress, max_files=limit)
                meta = meta_common | {"totals": totals}
                out_path = save_compact_graph_v2(v2_dir, meta, compact, gzip_out=gzip_out)
                st.success("Compact v2 graph built and saved!")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Files", totals["files"]); c2.metric("Classes", totals["classes"])
                c3.metric("Functions", totals["functions"]); c4.metric("Imports", totals["imports"])
                st.code(out_path, language="bash")
                if out_path.endswith(".gz"):
                    with gzip.open(out_path, "rt", encoding="utf-8") as f:
                        content = f.read()
                else:
                    with open(out_path, "r", encoding="utf-8") as f:
                        content = f.read()
                st.download_button("Download Compact Graph (v2)", content,
                                   file_name=os.path.basename(out_path),
                                   mime="application/json", use_container_width=True)

                if do_shard:
                    st.write("Creating shards (per top-level directory)...")
                    out_dir = os.path.join(v2_dir, meta["graph_id"])
                    manifest = shard_compact_by_top_dir(compact | {"meta": meta}, out_dir=out_dir, gzip_out=gzip_out)
                    man_path = os.path.join(out_dir, "manifest.json" + (".gz" if gzip_out else ""))
                    st.success(f"Manifest & shards written under: {out_dir}")
                    st.code(man_path, language="bash")

            # Cleanup temp
            try:
                shutil.rmtree(os.path.dirname(repo_root))
            except Exception:
                pass

            st.info("Next: use **Preview/Viewer** to explore, or **Generate Wiki** to create documentation XML.")

        except Exception as e:
            st.error(f"Error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

# ===================== Preview tab (per-file unified) =====================
with tab_preview:
    st.subheader("Explore a cached graph (per-file view)")
    try:
        mode = st.radio("Select source type", ["Verbose v1", "Compact v2 (single)", "Manifest (sharded)"], index=1, horizontal=True)
        base_dir = v2_dir if mode != "Verbose v1" else v1_dir

        ensure_dir(base_dir)
        # collect candidate files
        files = []
        if mode == "Verbose v1":
            files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.lower().endswith(".json")]
        elif mode == "Compact v2 (single)":
            files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".json") or f.endswith(".json.gz")]
        else:
            # list manifests under <v2>/<graph_id>/manifest.json[.gz]
            for sub in os.listdir(base_dir):
                man1 = os.path.join(base_dir, sub, "manifest.json")
                man2 = man1 + ".gz"
                if os.path.isfile(man1):
                    files.append(man1)
                if os.path.isfile(man2):
                    files.append(man2)

        files.sort(reverse=True)
        if not files:
            st.info("No files found. Build graphs first.")
        else:
            display = [os.path.relpath(p, base_dir) for p in files]
            selected_rel = st.selectbox("Select a graph source", options=display, key="preview_select")
            selected = os.path.join(base_dir, selected_rel)

            # Load and normalize to v1-like
            if mode == "Verbose v1":
                data_v1 = load_json_autoz(selected)
            elif mode == "Compact v2 (single)":
                compact = load_json_autoz(selected)
                data_v1 = expand_compact_to_v1(compact)
            else:
                manifest = load_json_autoz(selected)
                # merge shards into a single v1-like structure (may be heavy)
                nodes_all, edges_all = [], []
                for s in manifest.get("shards", []):
                    spath = s.get("path")
                    if spath and not os.path.isabs(spath):
                        spath = os.path.join(os.path.dirname(selected), spath)
                    shard = load_json_autoz(spath)
                    v1 = expand_compact_to_v1(shard)
                    nodes_all.extend(v1.get("nodes", []))
                    edges_all.extend(v1.get("edges", []))
                data_v1 = {"meta": manifest.get("meta", {}), "nodes": nodes_all, "edges": edges_all}

            meta = data_v1.get("meta", {})
            nodes = data_v1.get("nodes", [])
            edges = data_v1.get("edges", [])

            st.write(
                f"**Repo:** `{meta.get('owner','')}/{meta.get('repo','')}`  |  "
                f"**Branch:** `{meta.get('branch','')}`  |  "
                f"**Subpath:** `{meta.get('subpath','') or '.'}`"
            )

            # Build file-centric index
            nodes_by_id = {n["id"]: n for n in nodes}
            files_by_id = {n["id"]: n for n in nodes if n.get("type") == "file"}

            file_index = {
                fid: {
                    "path": fnode.get("path", fnode.get("label", "")),
                    "lang": fnode.get("lang", ""),
                    "classes": [],
                    "functions": [],
                    "imports": [],
                }
                for fid, fnode in files_by_id.items()
            }

            for e in edges:
                src = e.get("source"); dst = e.get("target"); etype = e.get("type")
                if src not in file_index: continue
                tnode = nodes_by_id.get(dst)
                if not tnode: continue

                if etype == "FILE_CONTAINS_CLASS" and tnode.get("type") == "class":
                    file_index[src]["classes"].append(tnode.get("label", ""))
                elif etype == "FILE_CONTAINS_FUNCTION" and tnode.get("type") == "function":
                    file_index[src]["functions"].append(tnode.get("label", ""))
                elif etype == "FILE_IMPORTS" and tnode.get("type") == "import":
                    file_index[src]["imports"].append(tnode.get("label", ""))

            for rec in file_index.values():
                rec["classes"] = sorted(set([x for x in rec["classes"] if x]))
                rec["functions"] = sorted(set([x for x in rec["functions"] if x]))
                rec["imports"] = sorted(set([x for x in rec["imports"] if x]))

            st.markdown("### Files (Imports, Classes, Functions)")

            colf1, colf2, colf3 = st.columns([2, 1, 1])
            with colf1:
                substr = st.text_input("Filter by file path (substring)", value="", key="file_filter")
                substr_norm = substr.strip().lower()
            with colf2:
                show_empty = st.checkbox("Show files with no detected symbols", value=False, key="show_empty_files")
            with colf3:
                view_mode = st.radio("View as", options=["Expanders", "Table"], horizontal=True, key="file_view_mode")

            sort_by = st.selectbox(
                "Sort by",
                options=["path", "lang", "imports", "classes", "functions"],
                index=0,
                key="sort_by_files"
            )
            desc = st.checkbox("Sort descending", value=False, key="sort_desc_files")

            rows = []
            for fid, rec in file_index.items():
                path = rec["path"] or files_by_id[fid].get("label", "")
                lang = rec.get("lang", "")
                is_empty = not (rec["classes"] or rec["functions"] or rec["imports"])
                if substr_norm and substr_norm not in path.lower(): continue
                if not show_empty and is_empty: continue
                rows.append({
                    "id": fid, "path": path, "lang": lang,
                    "imports": rec["imports"], "classes": rec["classes"], "functions": rec["functions"],
                    "imports_count": len(rec["imports"]), "classes_count": len(rec["classes"]), "functions_count": len(rec["functions"]),
                })

            key_map = {
                "path": lambda r: r["path"].lower(),
                "lang": lambda r: r.get("lang", ""),
                "imports": lambda r: r["imports_count"],
                "classes": lambda r: r["classes_count"],
                "functions": lambda r: r["functions_count"],
            }
            rows.sort(key=key_map[sort_by], reverse=desc)

            if not rows:
                st.info("No files match the current filters.")
            else:
                if view_mode == "Table":
                    try:
                        import pandas as pd
                        df = pd.DataFrame([{
                            "File": r["path"],
                            "Lang": r.get("lang", ""),
                            "Imports (#)": r["imports_count"],
                            "Classes (#)": r["classes_count"],
                            "Functions (#)": r["functions_count"],
                            "Imports": ", ".join(r["imports"]) if r["imports"] else "",
                            "Classes": ", ".join(r["classes"]) if r["classes"] else "",
                            "Functions": ", ".join(r["functions"]) if r["functions"] else "",
                        } for r in rows])
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "Download table (CSV)",
                            df.to_csv(index=False).encode("utf-8"),
                            file_name="files_summary.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    except Exception:
                        st.table([{
                            "File": r["path"], "Lang": r.get("lang", ""),
                            "Imports (#)": r["imports_count"], "Classes (#)": r["classes_count"], "Functions (#)": r["functions_count"],
                        } for r in rows])
                else:
                    for r in rows:
                        header = (
                            f"{r['path']}  —  lang: {r.get('lang','unknown')}  |  "
                            f"I:{r['imports_count']}  C:{r['classes_count']}  F:{r['functions_count']}"
                        )
                        with st.expander(header, expanded=False):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.markdown("**Imports**")
                                st.code("\n".join(r["imports"]) or "(none)")
                            with c2:
                                st.markdown("**Classes**")
                                st.code("\n".join(r["classes"]) or "(none)")
                            with c3:
                                st.markdown("**Functions**")
                                st.code("\n".join(r["functions"]) or "(none)")

            with st.expander("Raw JSON (first 300 lines)"):
                # Show the underlying (expanded) v1 JSON currently in use
                text = json.dumps(data_v1, indent=2, ensure_ascii=False).splitlines()
                st.code("\n".join(text[:300]))
    except Exception as e:
        st.error(f"Failed to load cached graphs: {e}")


# ===================== Generate Wiki (Single + Sharded) =====================

# Prompt builders for single-pass and sharded pipelines
XML_SPEC = """Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <sections>
    <section id="section-1">
      <title>[Section title]</title>
      <pages>
        <page_ref>page-1</page_ref>
      </pages>
      <subsections>
        <section_ref>section-2</section_ref>
      </subsections>
    </section>
  </sections>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
      </relevant_files>
      <related_pages>
        <related>page-2</related>
      </related_pages>
      <parent_section>section-1</parent_section>
    </page>
  </pages>
</wiki_structure>
"""

def summarize_for_prompt_v1like(graph_v1: dict, max_files_in_prompt: int = 200, top_imports_k: int = 50) -> dict:
    nodes = graph_v1.get("nodes", [])
    edges = graph_v1.get("edges", [])
    meta = graph_v1.get("meta", {})

    nodes_by_id = {n["id"]: n for n in nodes}
    file_nodes = [n for n in nodes if n.get("type") == "file"]

    file_index = {
        fn["id"]: {"path": fn.get("path", fn.get("label", "")), "lang": fn.get("lang", ""),
                   "classes": [], "functions": [], "imports": []}
        for fn in file_nodes
    }
    for e in edges:
        src = e.get("source"); dst = e.get("target"); etype = e.get("type")
        if src not in file_index: continue
        t = nodes_by_id.get(dst, {})
        if etype == "FILE_CONTAINS_CLASS" and t.get("type") == "class":
            file_index[src]["classes"].append(t.get("label", ""))
        elif etype == "FILE_CONTAINS_FUNCTION" and t.get("type") == "function":
            file_index[src]["functions"].append(t.get("label", ""))
        elif etype == "FILE_IMPORTS" and t.get("type") == "import":
            file_index[src]["imports"].append(t.get("label", ""))

    files = []
    langs = Counter()
    imports_counter = Counter()
    for rec in file_index.values():
        path = rec["path"]; lang = rec.get("lang") or "unknown"
        classes = sorted(set(rec["classes"]))
        functions = sorted(set(rec["functions"]))
        imps = sorted(set(rec["imports"]))
        langs[lang] += 1
        imports_counter.update(imps)
        files.append({"path": path, "lang": lang, "classes": classes, "functions": functions, "imports": imps,
                      "symbol_count": len(classes) + len(functions) + len(imps)})

    files.sort(key=lambda r: (r["symbol_count"], r["path"].lower()), reverse=True)
    files_included = files[:max_files_in_prompt]
    top_dirs = Counter()
    for f in files:
        parts = f["path"].split("/")
        if len(parts) > 1: top_dirs[parts[0]] += 1

    return {
        "meta": meta,
        "langs": langs.most_common(),
        "top_dirs": top_dirs.most_common(20),
        "top_imports": imports_counter.most_common(top_imports_k),
        "files": files_included,
        "files_total": len(files),
    }

def build_prompt_single(summary: dict, lang_text: str) -> str:
    meta = summary.get("meta", {})
    repo_id = f"{meta.get('owner','')}/{meta.get('repo','')}".strip("/")
    files_lines = []
    for f in summary["files"]:
        files_lines.append({
            "path": f["path"], "lang": f["lang"],
            "classes": f["classes"][:12], "functions": f["functions"][:12], "imports": f["imports"][:12],
        })
    return f"""You are a senior documentation architect.

I want to create a wiki for this repository: {repo_id or '(unknown)'}
Source: {meta.get('source_url', '(unknown)')}

IMPORTANT: The wiki content will be generated in {lang_text} language.

When designing the wiki structure, include pages that would benefit from visual diagrams:
- Architecture overviews
- State machines
- Class hierarchies

Create a structured wiki with these main sections (omit non-applicable, add justified ones):
- Overview
- System Architecture
- Core Features
- Data Management/Flow
- Frontend Components
- Backend Systems
- Model Integration
- Deployment/Infrastructure
- Extensibility and Customization

Guidelines:
- Use repository structure, languages, imports to infer boundaries.
- Group related pages and add related_pages.
- Assign importance per page (high/medium/low).
- Include <relevant_files> (paths) for each page.
- Return only the XML as per schema.

Repository summary:
- Totals: {json.dumps(summary.get('meta', {}).get('totals', {}), ensure_ascii=False)}
- Languages: {summary.get('langs', [])}
- Top-level directories: {summary.get('top_dirs', [])}
- Top imports: {summary.get('top_imports', [])}
- Files (up to {len(files_lines)} of {summary.get('files_total', 0)}):
{json.dumps(files_lines, ensure_ascii=False, indent=2)}

{XML_SPEC}

STRICT:
- Output ONLY <wiki_structure> XML. Use {lang_text}.
"""

# Sharded prompts
PARTIAL_XML_SPEC = """Return ONLY this exact structure for this shard:

<partial_wiki>
  <sections>
    <section id="sec-SHARD-1">
      <title>[Section title]</title>
      <pages>
        <page id="page-SHARD-1">
          <title>[Page title]</title>
          <description>[Brief description]</description>
          <importance>high|medium|low</importance>
          <relevant_files>
            <file_path>[path/to/file]</file_path>
          </relevant_files>
          <related_pages>
            <related>page-SHARD-2</related>
          </related_pages>
        </page>
      </pages>
      <subsections>
        <section_ref>sec-SHARD-2</section_ref>
      </subsections>
    </section>
  </sections>
</partial_wiki>
"""

def build_prompt_shard(owner_repo: str, source_url: str, lang_text: str, shard_name: str, shard_summary: dict) -> str:
    files = shard_summary["files"]
    files_lines = []
    for f in files:
        files_lines.append({
            "path": f["path"], "lang": f["lang"],
            "classes": f["classes"][:10], "functions": f["functions"][:10], "imports": f["imports"][:10],
        })
    return f"""You are a senior documentation architect.

We are creating a wiki for repository: {owner_repo or '(unknown)'}
Source: {source_url or '(unknown)'}
This prompt is for shard: {shard_name}

IMPORTANT: The wiki content will be generated in {lang_text} language.

Task:
- Propose sections and pages ONLY for this shard.
- Include relevant_files with file paths from this shard.
- Use IDs namespaced with the shard: sec-{shard_name}-N, page-{shard_name}-N.

Shard summary:
- Languages (by file count): {shard_summary.get('langs', [])}
- Files (up to {len(files_lines)} of {shard_summary.get('files_total', 0)}):
{json.dumps(files_lines, ensure_ascii=False, indent=2)}

{PARTIAL_XML_SPEC}
"""
def build_prompt_single_guarded(summary: dict, lang_text: str, allowed_sections_norm: list[str], forbidden_sections_norm: list[str], signals: dict) -> str:
    meta = summary.get("meta", {})
    repo_id = f"{meta.get('owner','')}/{meta.get('repo','')}".strip("/")
    files_lines = []
    for f in summary["files"]:
        files_lines.append({
            "path": f["path"], "lang": f["lang"],
            "classes": f["classes"][:12], "functions": f["functions"][:12], "imports": f["imports"][:12],
        })
    return f"""You are a senior documentation architect.

I want to create a wiki for this repository: {repo_id or '(unknown)'}
Source: {meta.get('source_url', '(unknown)')}

IMPORTANT: The wiki content will be generated in {lang_text} language.

## Evidence-derived constraints
- Allowed section titles (must ONLY use from this list, case-insensitive): {allowed_sections_norm}
- Forbidden section titles (do NOT use): {forbidden_sections_norm}
- Each page MUST include at least one valid <file_path> from the repository file list below.
- Do NOT invent content or placeholder sections that are not supported by repository evidence.

## Repository summary
- Totals: {json.dumps(summary.get('meta', {}).get('totals', {}), ensure_ascii=False)}
- Languages: {summary.get('langs', [])}
- Top-level directories: {summary.get('top_dirs', [])}
- Top imports: {summary.get('top_imports', [])}
- Files (up to {len(files_lines)} of {summary.get('files_total', 0)} total):
{json.dumps(files_lines, ensure_ascii=False, indent=2)}

{XML_SPEC}

STRICT:
- Output ONLY <wiki_structure> XML. Use {lang_text}.
- Section titles MUST be chosen from the allowed list and NOT appear in the forbidden list.
- Drop any section with no valid pages.
"""

def build_prompt_shard_guarded(owner_repo: str, source_url: str, lang_text: str, shard_name: str,
                               shard_summary: dict, allowed_sections_norm: list[str], forbidden_sections_norm: list[str]) -> str:
    files = shard_summary["files"]
    files_lines = []
    for f in files:
        files_lines.append({
            "path": f["path"], "lang": f["lang"],
            "classes": f["classes"][:10], "functions": f["functions"][:10], "imports": f["imports"][:10],
        })
    return f"""You are a senior documentation architect.

Repository: {owner_repo or '(unknown)'}
Source: {source_url or '(unknown)'}
This prompt is for shard: {shard_name}

IMPORTANT: The wiki content will be generated in {lang_text} language.

## Evidence-derived constraints
- Allowed section titles (must ONLY use from this list, case-insensitive): {allowed_sections_norm}
- Forbidden section titles: {forbidden_sections_norm}
- Each page MUST include at least one valid <file_path> from the shard file list below.
- Use IDs namespaced with the shard: sec-{shard_name}-N, page-{shard_name}-N.

Shard summary:
- Languages (by file count): {shard_summary.get('langs', [])}
- Files (up to {len(files_lines)} of {shard_summary.get('files_total', 0)}):
{json.dumps(files_lines, ensure_ascii=False, indent=2)}

{PARTIAL_XML_SPEC}

STRICT:
- Output ONLY <partial_wiki>.
- Use only allowed section titles; drop forbidden ones.
"""

def summarize_shard_files(files: List[Dict[str, Any]], dict_imports: List[str], max_files: int = 200) -> dict:
    recs = []
    for f in files:
        imps = [dict_imports[i] for i in f.get("imports", []) if 0 <= i < len(dict_imports)]
        sym = len(f.get("classes", [])) + len(f.get("functions", [])) + len(imps)
        recs.append({"path": f["path"], "lang": f.get("lang") or "unknown",
                     "classes": f.get("classes", []), "functions": f.get("functions", []),
                     "imports": imps, "symbol_count": sym})
    recs.sort(key=lambda r: (r["symbol_count"], r["path"].lower()), reverse=True)
    return {"files": recs[:max_files], "files_total": len(files), "langs": Counter([r["lang"] for r in recs]).most_common()}

def extract_partial(xml_text: str) -> Any:
    import xml.etree.ElementTree as ET
    s = xml_text.strip()
    start = s.find("<partial_wiki")
    end = s.rfind("</partial_wiki>")
    if start == -1 or end == -1:
        s = f"<partial_wiki>{s}</partial_wiki>"
        start = 0; end = len(s) - len("</partial_wiki>")
    frag = s[start:end + len("</partial_wiki>")]
    try:
        return ET.fromstring(frag)
    except Exception:
        return ET.fromstring("<partial_wiki><sections/></partial_wiki>")

def normalize_title(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = s.replace("&", "and")
    return s

def merge_partials(partials: List[Any]) -> Any:
    import xml.etree.ElementTree as ET
    section_by_key = {}
    page_pool = []

    for frag in partials:
        sects = frag.find("sections")
        if sects is None: continue
        for s in list(sects.findall("section")):
            title = (s.findtext("title") or "").strip()
            if not title: continue
            key = normalize_title(title)
            entry = section_by_key.setdefault(key, {"title": title, "pages": []})
            # pages
            pages = s.find("pages")
            if pages is not None:
                for p in list(pages.findall("page")):
                    page = {
                        "id": p.get("id") or "",
                        "title": (p.findtext("title") or "").strip(),
                        "description": (p.findtext("description") or "").strip(),
                        "importance": (p.findtext("importance") or "medium").strip(),
                        "relevant_files": [fp.text.strip() for fp in p.findall("./relevant_files/file_path") if (fp.text or "").strip()],
                        "related_pages": [rp.text.strip() for rp in p.findall("./related_pages/related") if (rp.text or "").strip()],
                        "parent_key": key
                    }
                    if page["title"]:
                        entry["pages"].append(page)
                        page_pool.append(page)

    root = ET.Element("root")
    sections_el = ET.SubElement(root, "sections")
    pages_el = ET.SubElement(root, "pages")

    sec_id_map = {}
    for i, (key, entry) in enumerate(section_by_key.items(), start=1):
        sid = f"section-{i}"
        sec_id_map[key] = sid
        s = ET.SubElement(sections_el, "section", {"id": sid})
        ET.SubElement(s, "title").text = entry["title"]

    page_id_map = {}
    for i, page in enumerate(page_pool, start=1):
        pid = f"page-{i}"
        page_id_map[page["id"]] = pid
        p = ET.SubElement(pages_el, "page", {"id": pid})
        ET.SubElement(p, "title").text = page["title"]
        ET.SubElement(p, "description").text = page["description"]
        ET.SubElement(p, "importance").text = page["importance"] or "medium"
        rf = ET.SubElement(p, "relevant_files")
        for fp in sorted(set(page["relevant_files"])):
            ET.SubElement(rf, "file_path").text = fp
        rp = ET.SubElement(p, "related_pages")
        for r in page.get("related_pages", []):
            if r in page_id_map:
                ET.SubElement(rp, "related").text = page_id_map[r]
            else:
                ET.SubElement(rp, "related").text = r
        ET.SubElement(p, "parent_section").text = sec_id_map.get(page.get("parent_key", ""), "section-1")

    for s in list(sections_el.findall("section")):
        title = s.findtext("title") or ""
        key = normalize_title(title)
        entry = section_by_key.get(key, {})
        pages = entry.get("pages", [])
        pages_container = ET.SubElement(s, "pages")
        for page in pages:
            final_id = page_id_map.get(page.get("id", ""))
            if final_id:
                ET.SubElement(pages_container, "page_ref").text = final_id

    return root

def compose_final_wiki(owner_repo: str, description: str, merged_root: Any) -> str:
    import xml.etree.ElementTree as ET
    ws = ET.Element("wiki_structure")
    ET.SubElement(ws, "title").text = f"Wiki – {owner_repo}" if owner_repo else "Wiki"
    ET.SubElement(ws, "description").text = description or "Auto-generated wiki structure."

    src_sections = merged_root.find("sections")
    dst_sections = ET.SubElement(ws, "sections")
    if src_sections is not None:
        for s in list(src_sections.findall("section")):
            dst_sections.append(s)

    src_pages = merged_root.find("pages")
    dst_pages = ET.SubElement(ws, "pages")
    if src_pages is not None:
        for p in list(src_pages.findall("page")):
            dst_pages.append(p)

    return ET.tostring(ws, encoding="unicode")

with tab_wiki:
    st.subheader("📄 Generate Wiki (Local HF Model)")
    try:
        pipeline = st.radio("Pipeline", ["Sharded (manifest, map→reduce)", "Single-pass (v1/v2)"], index=0, horizontal=True)
        lang_text = st.text_input("Target language", "English")
        strict_relevance = st.checkbox("Strict relevance mode (guarded prompt + pruning)", value=True)
        # colA, colB = st.columns(2)
        # with colA:
        #     model_id_or_path = st.text_input("Model ID or local path", "microsoft/Phi-3.5-mini-instruct")
            
        #     hf_cache_dir = st.text_input("HF cache dir (optional)", value="")
        #     quant_choice = st.radio("Quantization", ["None", "8-bit", "4-bit"], index=0, horizontal=True)
        #     pre_dl = st.checkbox("Pre-download with AutoModel", value=False)
        # with colB:
        #     max_new_tokens = st.slider("Max new tokens", 128, 4096, 900, step=64)
        #     do_sample = st.checkbox("Enable sampling", value=False)
        #     temperature = st.slider("Temperature", 0.0, 1.5, 0.2, step=0.05)
        #     top_p = st.slider("Top-p", 0.1, 1.0, 0.95, step=0.05)
        #     repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.05, step=0.01)

        if pipeline == "Single-pass (v1/v2)":
            mode = st.radio("Source type", ["Verbose v1", "Compact v2 (single)"], index=1, horizontal=True, key="wiki_single_mode")
            base_dir = v2_dir if mode == "Compact v2 (single)" else v1_dir
            ensure_dir(base_dir)

            candidates = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                          if (f.endswith(".json") or f.endswith(".json.gz"))]
            candidates.sort(reverse=True)
            if not candidates:
                st.info("No graphs found. Build one first.")
                st.stop()
            display = [os.path.relpath(p, base_dir) for p in candidates]
            selected_rel = st.selectbox("Select graph file", options=display, key="wiki_single_select")
            selected = os.path.join(base_dir, selected_rel)

            max_files_in_prompt = st.number_input("Max files in prompt", 20, 1000, 200, step=10)
            top_imports_k = st.slider("Top-K imports", 10, 200, 50, step=5)

            run = st.button("Generate Wiki XML (Single-pass)", type="primary", use_container_width=True)
            if run:
                try:

                    if mode == "Verbose v1":
                        data_v1 = load_json_autoz(selected)
                        all_paths = _collect_all_paths_from_v1like(data_v1)
                        summary = summarize_for_prompt_v1like(data_v1, max_files_in_prompt=int(max_files_in_prompt), top_imports_k=top_imports_k)
                    else:
                        compact = load_json_autoz(selected)
                        data_v1 = expand_compact_to_v1(compact)
                        all_paths = _collect_all_paths_from_compact(compact)
                        summary = summarize_for_prompt_v1like(compact, max_files_in_prompt=int(max_files_in_prompt), top_imports_k=top_imports_k)

                    signals = _derive_repo_signals_from_paths(all_paths)
                    allowed_norm, forbidden_norm = _allowed_forbidden_sections(signals)

                    if strict_relevance:
                        prompt = build_prompt_single_guarded(summary, lang_text, allowed_norm, forbidden_norm, signals)
                    else:
                        prompt = build_prompt_single(summary, lang_text)

                    
                    prompt = build_prompt_single(summary, lang_text=lang_text)
                    xml_text = qgenie_generate_xml(
                        prompt_text=prompt
                    )
                    xml_text = validate_or_wrap_xml(xml_text, "wiki_structure")

                    if strict_relevance:
                        xml_text = _prune_and_renumber_wiki(xml_text, allowed_norm, set(all_paths))
                    # Determine correct meta for naming
                    graph_meta_for_save = {}
                    if mode == "Verbose v1":
                        graph_meta_for_save = data_v1.get("meta", {})
                    else:
                        # We loaded compact above, keep its meta
                        graph_meta_for_save = compact.get("meta", {})
                    
                    # Save using the standard path
                    final_path = wiki_default_output_path(graph_meta_for_save)
                    ensure_dir(os.path.dirname(final_path))
                    with open(final_path, "w", encoding="utf-8") as f:
                        f.write(xml_text)
                    
                    st.success("Wiki generation complete ✅")
                    st.write("**Saved final XML:**")
                    st.code(final_path, language="bash")
                    
                    # Preview + download
                    st.code(xml_text, language="xml")
                    st.download_button(
                        "Download Wiki XML",
                        data=xml_text,
                        file_name=os.path.basename(final_path),
                        mime="application/xml",
                        use_container_width=True,
                    )
                    
                except Exception as e:
                    st.error(f"Failed: {e}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

        else:
            # Sharded pipeline
            ensure_dir(v2_dir)
            manifests = []
            for sub in os.listdir(v2_dir):
                man1 = os.path.join(v2_dir, sub, "manifest.json")
                man2 = man1 + ".gz"
                if os.path.isfile(man1): manifests.append(man1)
                if os.path.isfile(man2): manifests.append(man2)
            manifests.sort(reverse=True)
            if not manifests:
                st.info("No manifests found. Build compact v2 with shards first.")
                st.stop()
            display = [os.path.relpath(m, v2_dir) for m in manifests]
            selected_rel = st.selectbox("Select manifest", options=display, key="wiki_manifest_select")
            manifest_path = os.path.join(v2_dir, selected_rel)

            max_files_per_shard = st.number_input("Max files per shard (prompt)", 20, 1000, 200, step=10)
            refine = st.checkbox("Final refine pass", value=True)
            refine_max_new_tokens = st.slider("Refine max new tokens", 128, 4096, 900, step=64)

            run_sharded = st.button("Generate Wiki XML (Sharded map→reduce)", type="primary", use_container_width=True)
            if run_sharded:
                try:

                    manifest = load_json_autoz(manifest_path)
                    graph_meta = manifest.get("meta", {})  # <-- use meta for naming
                    owner_repo = f"{graph_meta.get('owner','')}/{graph_meta.get('repo','')}".strip("/")
                    source_url = graph_meta.get("source_url", "")

                    all_paths = []
                    for s in manifest.get("shards", []):
                        spath = s.get("path")
                        if spath and not os.path.isabs(spath):
                            spath = os.path.join(os.path.dirname(manifest_path), spath)
                        shard = load_json_autoz(spath)
                        all_paths.extend(_collect_all_paths_from_compact(shard))
                    signals = _derive_repo_signals_from_paths(all_paths)
                    allowed_norm, forbidden_norm = _allowed_forbidden_sections(signals)

                    # MAP: per shard (use guarded shard prompt if strict)
                    partial_xmls = []
                    saved_partial_paths = []
                    for idx, s in enumerate(manifest.get("shards", []), start=1):
                        spath = s.get("path")
                        if spath and not os.path.isabs(spath):
                            spath = os.path.join(os.path.dirname(manifest_path), spath)
                        shard = load_json_autoz(spath)

                        shard_name = shard.get("meta", {}).get("shard") or f"shard{idx}"
                        safe_shard_name = shard_name.replace("/", "_")
                        dict_imports = shard.get("dicts", {}).get("imports", [])
                        files = shard.get("files", [])
                        shard_summary = summarize_shard_files(files, dict_imports, max_files=int(max_files_per_shard))

                        if strict_relevance:
                            prompt = build_prompt_shard_guarded(owner_repo, source_url, lang_text, safe_shard_name, shard_summary, allowed_norm, forbidden_norm)
                        else:
                            prompt = build_prompt_shard(owner_repo, source_url, lang_text, safe_shard_name, shard_summary)

                        xml = qgenie_generate_xml(
                            prompt_text=prompt,
                        )

                        # Save each partial
                        partial_path = wiki_partial_output_path(graph_meta, safe_shard_name)
                        ensure_dir(os.path.dirname(partial_path))
                        with open(partial_path, "w", encoding="utf-8") as f:
                            f.write(xml)
                        saved_partial_paths.append(partial_path)
                        partial_xmls.append(xml)

                    # REDUCE
                    partial_trees = [extract_partial(x) for x in partial_xmls]
                    merged_root = merge_partials(partial_trees)
                    final_xml = compose_final_wiki(owner_repo=owner_repo, description="Auto-generated wiki structure.", merged_root=merged_root)

                    # PRUNE (strict)
                    if strict_relevance:
                        final_xml = _prune_and_renumber_wiki(final_xml, allowed_norm, set(all_paths))

                    # REFINE (optional)
                    if refine:
                        refine_prompt = f"""You are a meticulous documentation architect.
                    We combined shard-level partial wikis into one structure for repository: {owner_repo or '(unknown)'}
                    Source: {source_url or '(unknown)'}

                    Please refine and normalize the merged XML below into a final, consistent <wiki_structure>:
                    - Ensure IDs are sequential (section-1..N, page-1..M).
                    - Ensure every <page> has exactly one <parent_section> and the parent exists.
                    - Ensure all <section_ref> and <page_ref> targets exist.
                    - Preserve page titles/descriptions but improve clarity/consistency.
                    - Keep it concise and clean.
                    - Use {lang_text}.
                    Merged XML:
                    {final_xml}

                    {XML_SPEC}
                    """
                        final_xml = qgenie_generate_xml(
                            prompt_text=refine_prompt
                        )
                        final_xml = validate_or_wrap_xml(final_xml, "wiki_structure")

                    # Save final using your standard naming
                    final_path = wiki_default_output_path(graph_meta)
                    ensure_dir(os.path.dirname(final_path))
                    with open(final_path, "w", encoding="utf-8") as f:
                        f.write(final_xml)

                    st.success("Sharded wiki generation complete ✅")
                    st.write("**Saved shard partials:**")
                    for p in saved_partial_paths:
                        st.code(p, language="bash")
                    st.write("**Saved final XML:**")
                    st.code(final_path, language="bash")
                    st.code(final_xml, language="xml")
                    st.download_button("Download Wiki XML", data=final_xml, file_name=os.path.basename(final_path),
                                    mime="application/xml", use_container_width=True)


                except Exception as e:
                    st.error(f"Failed: {e}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())


    except Exception as e:
        st.error(f"Failed: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
