import ast
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from urllib.request import Request, urlopen
import streamlit as st
import streamlit.components.v1 as components

# Optional: PyVis (for interactive graph)
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except Exception:
    PYVIS_AVAILABLE = False

# ===================== Configuration =====================
DEFAULT_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "graphs")
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

SKIP_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "build", "dist", "node_modules",
    ".mypy_cache", ".pytest_cache", "target", "out", ".gradle", ".idea"
}
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2MB per file safety limit

# Language sets by extension (lowercase, without dot)
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

def http_get_json(url: str, token: str | None = None):
    headers = {"User-Agent": "streamlit-kg-app"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_default_branch(owner: str, repo: str, token: str | None = None) -> str:
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

def iter_repo_files(root_dir: str, subpath: str | None = None, max_files: int | None = None):
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

def read_text_file(path: str) -> str | None:
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
    for m in re.finditer(r"""import\s+(?:[\s\S]*?\s+from\s+)?['"'"]""", src):
        imports.append({"type": "import", "module": m.group(1), "names": [], "level": 0})
    for m in re.finditer(r"""require\(\s*'"['"]\s*\)""", src):
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
    for m in re.finditer(r"\bimport\s+([a-zA-Z0-9_\.]+)(?:\s*;\s*)", src):
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

# ===================== Graph Build =====================

def build_graph_struct():
    return {"nodes": [], "edges": []}

def add_node(nodes_index, graph, node_id, node_type, label, **props):
    if node_id in nodes_index:
        return
    node = {"id": node_id, "type": node_type, "label": label}
    node.update(props)
    graph["nodes"].append(node)
    nodes_index[node_id] = node

def add_edge(graph, source, target, edge_type, **props):
    edge = {"source": source, "target": target, "type": edge_type}
    edge.update(props)
    graph["edges"].append(edge)

def build_repo_kg(root_dir: str, subpath: str | None = None, progress=None, max_files: int | None = None):
    """
    Build knowledge graph across ALL files:
      - file nodes (with language guess from extension)
      - class nodes and FILE_CONTAINS_CLASS edges
      - function nodes and FILE_CONTAINS_FUNCTION edges
      - import nodes (module/header string) and FILE_IMPORTS edges
    """
    graph = build_graph_struct()
    nodes_index = {}
    totals = {"files": 0, "classes": 0, "imports": 0, "functions": 0}

    file_list = list(iter_repo_files(root_dir, subpath, max_files=max_files))
    n = len(file_list)

    for i, (abs_path, rel_path) in enumerate(file_list, start=1):
        if progress:
            progress.progress(min(i / max(n, 1), 1.0), text=f"Scanning {rel_path} ({i}/{n})")

        ext = os.path.splitext(rel_path)[1]
        lang = detect_lang_by_ext(ext)
        file_text = read_text_file(abs_path)

        file_node_id = f"file:{rel_path}"
        add_node(nodes_index, graph, file_node_id, "file", rel_path, path=rel_path, lang=lang)
        totals["files"] += 1

        if not file_text:
            continue

        classes, imports, functions = parse_any_text_by_ext(ext, file_text)

        for cls in classes:
            class_node_id = f"class:{rel_path}#{cls}"
            add_node(nodes_index, graph, class_node_id, "class", cls, file=rel_path, lang=lang)
            add_edge(graph, file_node_id, class_node_id, "FILE_CONTAINS_CLASS")
        totals["classes"] += len(classes)

        for fn in functions:
            func_node_id = f"function:{rel_path}#{fn}"
            add_node(nodes_index, graph, func_node_id, "function", fn, file=rel_path, lang=lang)
            add_edge(graph, file_node_id, func_node_id, "FILE_CONTAINS_FUNCTION")
        totals["functions"] += len(functions)

        for imp in imports:
            module = imp.get("module") or ""
            if not module:
                continue
            import_node_id = f"import:{module}"
            add_node(nodes_index, graph, import_node_id, "import", module, kind=imp.get("type"))
            add_edge(graph, file_node_id, import_node_id, "FILE_IMPORTS", detail=imp.get("type"))
        totals["imports"] += len(imports)

    return graph, totals

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def fingerprint(owner: str, repo: str, branch: str, subpath: str | None, url: str) -> str:
    payload = f"{owner}/{repo}@{branch}:{subpath or ''}|{url}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

def save_graph(cache_dir: str, meta: dict, graph: dict) -> str:
    ensure_dir(cache_dir)
    graph_id = meta["graph_id"]
    filename = f"{meta['owner']}__{meta['repo']}__{meta['branch']}__{graph_id}.json"
    filename = filename.replace("/", "_")
    out_path = os.path.join(cache_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, **graph}, f, ensure_ascii=False, indent=2)
    return out_path

# ===================== Streamlit UI =====================

st.set_page_config(page_title="GitHub → Knowledge Graph (Polyglot + Viewer)", layout="wide")
st.title("📚 GitHub → Knowledge Graph (Polyglot: files, classes, imports, functions) + 🕸️ Interactive Viewer")

with st.sidebar:
    st.header("Input")
    gh_url = st.text_input(
        "GitHub URL",
        help="Examples:\n- https://github.com/pallets/flask\n- https://github.com/pallets/flask/tree/main/src\n- https://github.com/owner/repo/blob/main/path/to/file.js"
    )
    token = st.text_input("GitHub token (optional)", type="password",
                          help="Improves API rate limits. Create a fine-grained token with minimal permissions.")
    cache_dir = st.text_input("Cache directory", value=DEFAULT_CACHE_DIR)
    max_files = st.number_input("Max files to scan (0 = no limit)", min_value=0, value=0, step=25)
    run_btn = st.button("Build Knowledge Graph", type="primary", use_container_width=True)

tab_build, tab_preview, tab_viewer = st.tabs(["🔨 Build", "👀 Preview / Explore", "🕸️ Graph Viewer (PyVis)"])

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

            st.write("Parsing repository files (polyglot)...")
            progress = st.progress(0.0, text="Starting...")
            limit = max_files if max_files and max_files > 0 else None
            graph, totals = build_repo_kg(repo_root, subpath=subpath, progress=progress, max_files=limit)

            g_id = fingerprint(owner, repo, branch, subpath, gh_url)
            meta = {
                "source_url": gh_url,
                "owner": owner,
                "repo": repo,
                "branch": branch,
                "subpath": subpath or "",
                "created_at": now_iso(),
                "totals": totals,
                "graph_id": g_id,
                "notes": "Non-Python languages parsed heuristically via regex; Python via AST."
            }

            saved_path = save_graph(cache_dir, meta, graph)

            try:
                shutil.rmtree(os.path.dirname(repo_root))
            except Exception:
                pass

            st.success("Knowledge graph built and saved!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Files", totals["files"])
            c2.metric("Classes", totals["classes"])
            c3.metric("Functions", totals["functions"])
            c4.metric("Imports", totals["imports"])

            st.code(saved_path, language="bash")
            with open(saved_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "Download Graph JSON",
                    f.read(),
                    file_name=os.path.basename(saved_path),
                    mime="application/json",
                    use_container_width=True,
                )

            st.info("Tip: switch to **🕸️ Graph Viewer (PyVis)** tab to interactively visualize any cached graph.")

        except Exception as e:
            st.error(f"Error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

# ===================== Preview tab (tables) =====================
with tab_preview:
    st.subheader("Explore a cached graph (per-file view)")
    try:
        ensure_dir(cache_dir)
        files = [f for f in os.listdir(cache_dir) if f.lower().endswith(".json")]
        files.sort(reverse=True)
        selected = st.selectbox("Select a cached graph", options=files, key="preview_select")
        if selected:
            full_path = os.path.join(cache_dir, selected)
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])

            st.write(
                f"**Repo:** `{meta.get('owner','')}/{meta.get('repo','')}`  |  "
                f"**Branch:** `{meta.get('branch','')}`  |  "
                f"**Subpath:** `{meta.get('subpath','') or '.'}`"
            )
            totals = meta.get("totals", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Files", totals.get("files", 0))
            c2.metric("Classes", totals.get("classes", 0))
            c3.metric("Functions", totals.get("functions", 0))
            c4.metric("Imports", totals.get("imports", 0))

            # ---------- Unified per-file view (Imports, Classes, Functions) ----------

            # Build file-centric index from nodes & edges
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
                src = e.get("source")
                dst = e.get("target")
                etype = e.get("type")
                if src not in file_index:
                    continue
                target_node = nodes_by_id.get(dst)
                if not target_node:
                    continue

                if etype == "FILE_CONTAINS_CLASS" and target_node.get("type") == "class":
                    file_index[src]["classes"].append(target_node.get("label", ""))
                elif etype == "FILE_CONTAINS_FUNCTION" and target_node.get("type") == "function":
                    file_index[src]["functions"].append(target_node.get("label", ""))
                elif etype == "FILE_IMPORTS" and target_node.get("type") == "import":
                    file_index[src]["imports"].append(target_node.get("label", ""))

            # De-duplicate + sort lists
            for rec in file_index.values():
                rec["classes"] = sorted(set([x for x in rec["classes"] if x]))
                rec["functions"] = sorted(set([x for x in rec["functions"] if x]))
                rec["imports"] = sorted(set([x for x in rec["imports"] if x]))

            st.markdown("### Files (Imports, Classes, Functions)")

            # Filters & view options
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

            # Build rows with filtering
            rows = []
            for fid, rec in file_index.items():
                path = rec["path"] or files_by_id[fid].get("label", "")
                lang = rec.get("lang", "")
                is_empty = not (rec["classes"] or rec["functions"] or rec["imports"])

                if substr_norm and substr_norm not in path.lower():
                    continue
                if not show_empty and is_empty:
                    continue

                rows.append({
                    "id": fid,
                    "path": path,
                    "lang": lang,
                    "imports": rec["imports"],
                    "classes": rec["classes"],
                    "functions": rec["functions"],
                    "imports_count": len(rec["imports"]),
                    "classes_count": len(rec["classes"]),
                    "functions_count": len(rec["functions"]),
                })

            # Sorting
            key_map = {
                "path": lambda r: r["path"].lower(),
                "lang": lambda r: r.get("lang", ""),
                "imports": lambda r: r["imports_count"],
                "classes": lambda r: r["classes_count"],
                "functions": lambda r: r["functions_count"],
            }
            rows.sort(key=key_map[sort_by], reverse=desc)

            # Render unified view
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
                        # Fallback if pandas isn't available
                        st.table([{
                            "File": r["path"],
                            "Lang": r.get("lang", ""),
                            "Imports (#)": r["imports_count"],
                            "Classes (#)": r["classes_count"],
                            "Functions (#)": r["functions_count"],
                        } for r in rows])
                else:
                    # Expanders (grouped by file)
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

            # Raw JSON excerpt
            with st.expander("Raw JSON (first 300 lines)"):
                text = json.dumps(data, indent=2, ensure_ascii=False).splitlines()
                st.code("\n".join(text[:300]))
    except Exception as e:
        st.error(f"Failed to load cached graphs: {e}")


# ===================== Graph Viewer (PyVis) =====================

def build_pyvis_html(data: dict,
                     include_types: set[str],
                     lang_filter: str | None,
                     path_contains: str | None,
                     max_nodes: int = 1000,
                     physics: bool = True,
                     hierarchical: bool = False,
                     height_px: int = 720) -> str:
    """
    Returns an HTML string for embedding the interactive graph.
    """
    if not PYVIS_AVAILABLE:
        raise RuntimeError("PyVis is not installed. Run: pip install pyvis")

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Map node_id -> node
    node_by_id = {n["id"]: n for n in nodes}

    # Filter nodes by type/lang/path
    def node_passes(n):
        if n["type"] not in include_types:
            return False
        if path_contains:
            if n["type"] == "file":
                if path_contains.lower() not in n.get("path", n.get("label","")).lower():
                    return False
            else:
                # for class/function nodes, filter by associated file path if available
                f = n.get("file")
                if f and path_contains.lower() not in f.lower():
                    return False
        if lang_filter and n.get("lang"):
            return n.get("lang") == lang_filter
        if lang_filter and n["type"] == "file":
            return n.get("lang") == lang_filter
        return True

    filtered_nodes = [n for n in nodes if node_passes(n)]

    # Limit node count to avoid browser overload
    if max_nodes and len(filtered_nodes) > max_nodes:
        filtered_nodes = filtered_nodes[:max_nodes]
    allowed_ids = set(n["id"] for n in filtered_nodes)

    # Filter edges to those whose both endpoints are included
    filtered_edges = [e for e in edges if e["source"] in allowed_ids and e["target"] in allowed_ids]

    # Degree for sizing
    degree = {nid: 0 for nid in allowed_ids}
    for e in filtered_edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    # Colors & shapes per type
    COLORS = {
        "file": "#1f77b4",      # blue
        "class": "#2ca02c",     # green
        "function": "#9467bd",  # purple
        "import": "#ff7f0e",    # orange
    }
    SHAPES = {
        "file": "box",
        "class": "dot",
        "function": "dot",
        "import": "diamond",
    }

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#222222", directed=False, notebook=False)
    net.toggle_physics(physics)

    # Add nodes
    for n in filtered_nodes:
        nid = n["id"]
        ntype = n["type"]
        color = COLORS.get(ntype, "#888888")
        shape = SHAPES.get(ntype, "dot")
        deg = degree.get(nid, 1)
        size = max(8, min(28, 8 + deg * 1.8))

        # Label & title (tooltip)
        if ntype == "file":
            label = os.path.basename(n.get("path", n.get("label", ""))) or n.get("label", "")
            title = f"<b>File</b><br>path: {n.get('path', n.get('label',''))}<br>lang: {n.get('lang','unknown')}"
        elif ntype in ("class", "function"):
            label = n.get("label", "")
            title = f"<b>{ntype.title()}</b><br>name: {label}<br>file: {n.get('file','')}<br>lang: {n.get('lang','unknown')}"
        else:  # import
            label = n.get("label", "")
            title = f"<b>Import</b><br>module: {label}<br>kind: {n.get('kind','')}"
        net.add_node(nid, label=label, title=title, color=color, shape=shape, size=size)

    # Edge styles by type
    EDGE_COLORS = {
        "FILE_CONTAINS_CLASS": "#7fbf7f",
        "FILE_CONTAINS_FUNCTION": "#b07fbf",
        "FILE_IMPORTS": "#ffbb78",
    }

    for e in filtered_edges:
        etype = e.get("type", "")
        color = EDGE_COLORS.get(etype, "#cccccc")
        net.add_edge(e["source"], e["target"], color=color, title=etype)

    # Options: physics + optional hierarchical layout
    options_dict = {
    "nodes": {
        "font": {"size": 12, "face": "Inter"}
    },
    "edges": {
        "smooth": False,
        "color": {"color": "#cccccc"}
    },
    "physics": {
        "enabled": bool(physics),
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -2000,
            "springLength": 120,
            "springConstant": 0.02
        }
    },
    "layout": {
        "hierarchical": {
            "enabled": bool(hierarchical),
            "direction": "LR",
            "sortMethod": "hubsize",
            "levelSeparation": 180,
            "nodeSpacing": 120
        }
    },
    "interaction": {
        "hover": True,
        "dragNodes": True,
        "zoomView": True,
        "navigationButtons": True,
        "keyboard": True
    }
}
    net.set_options(json.dumps(options_dict))

    # Return HTML
    try:
        return net.generate_html()
    except Exception:
        # Fallback: write temp file then read back
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.close()
        net.show(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp.name)
        return html

with tab_viewer:
    st.subheader("Interactive Network Graph (PyVis)")
    if not PYVIS_AVAILABLE:
        st.warning("PyVis is not installed. Install it with: `pip install pyvis`")
    else:
        try:
            ensure_dir(cache_dir)
            files = [f for f in os.listdir(cache_dir) if f.lower().endswith(".json")]
            files.sort(reverse=True)
            selected_viz = st.selectbox("Select a cached graph to visualize", options=files, key="viewer_select")
            if selected_viz:
                full_path_viz = os.path.join(cache_dir, selected_viz)
                with open(full_path_viz, "r", encoding="utf-8") as f:
                    data_viz = json.load(f)

                # Controls
                st.markdown("#### Filters & Layout")
                colA, colB, colC = st.columns([1.3, 1, 1])
                with colA:
                    include_types = st.multiselect(
                        "Node types",
                        options=["file", "class", "function", "import"],
                        default=["file", "class", "function", "import"]
                    )
                with colB:
                    # languages present in file nodes
                    langs = sorted({n.get("lang") for n in data_viz.get("nodes", []) if n.get("lang")})
                    lang_filter = st.selectbox("Filter by language (files)", options=["(all)"] + langs)
                    lang_filter = None if lang_filter == "(all)" else lang_filter
                with colC:
                    path_contains = st.text_input("Path contains (substring)", value="", placeholder="e.g., src/ or utils")
                    path_contains = path_contains.strip() or None

                colD, colE, colF = st.columns([1, 1, 1])
                with colD:
                    max_nodes = st.slider("Max nodes to render", min_value=100, max_value=5000, value=1200, step=100)
                with colE:
                    physics = st.toggle("Physics (Barnes–Hut)", value=True)
                with colF:
                    hierarchical = st.toggle("Hierarchical layout (LR)", value=False)

                render = st.button("Render Graph", type="primary", use_container_width=True)

                if render:
                    try:
                        html = build_pyvis_html(
                            data=data_viz,
                            include_types=set(include_types),
                            lang_filter=lang_filter,
                            path_contains=path_contains,
                            max_nodes=max_nodes,
                            physics=physics,
                            hierarchical=hierarchical,
                            height_px=720,
                        )
                        components.html(html, height=760, scrolling=True)

                        st.download_button(
                            "Download Visualization as HTML",
                            data=html,
                            file_name=os.path.splitext(selected_viz)[0] + "_graph.html",
                            mime="text/html",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Failed to render graph: {e}")
                        with st.expander("Traceback"):
                            st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"Failed to load cached graphs: {e}")
