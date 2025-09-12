#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a documentation wiki XML structure from a cached repo knowledge graph JSON
using a local Hugging Face model via transformers (no HF Inference API).

Examples:

# Basic (GPU if available)
python wiki_from_graph_local.py \
  --graph ./.cache/graphs/owner__repo__main__a1b2c3d4e5f6.json \
  --lang-text "English" \
  --model "meta-llama/Llama-3.1-8B-Instruct"

# Use a local model folder already downloaded to disk
python wiki_from_graph_local.py \
  --graph ./.cache/graphs/owner__repo__main__a1b2c3d4e5f6.json \
  --lang-text "English" \
  --model /path/to/local/model_dir

# Pre-download via AutoModel (to satisfy policy), then load LM head for generation
python wiki_from_graph_local.py \
  --graph ./.cache/graphs/owner__repo__main__a1b2c3d4e5f6.json \
  --lang-text "English" \
  --model "mistralai/Mistral-7B-Instruct-v0.3" \
  --pre-download-with-automodel

# Advanced generation controls + stop at closing tag
python wiki_from_graph_local.py \
  --graph ./.cache/graphs/owner__repo__main__a1b2c3d4e5f6.json \
  --lang-text "English" \
  --model "mistralai/Mistral-7B-Instruct-v0.3" \
  --max-new-tokens 1200 --temperature 0.2 --top-p 0.95 --do-sample \
  --eos-string "</wiki_structure>"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# ---------------------- Graph loading & summarization ----------------------

def load_graph(graph_path: str) -> Dict[str, Any]:
    if not os.path.isfile(graph_path):
        raise FileNotFoundError(f"Graph JSON not found: {graph_path}")
    with open(graph_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_file_index(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      file_index: file_node_id -> {
        'path': str, 'lang': str, 'classes': [str], 'functions': [str], 'imports': [str]
      }
    """
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
        if src not in file_index:
            continue
        tnode = nodes_by_id.get(dst)
        if not tnode:
            continue
        ttype = tnode.get("type")
        label = tnode.get("label", "")

        if etype == "FILE_CONTAINS_CLASS" and ttype == "class":
            file_index[src]["classes"].append(label)
        elif etype == "FILE_CONTAINS_FUNCTION" and ttype == "function":
            file_index[src]["functions"].append(label)
        elif etype == "FILE_IMPORTS" and ttype == "import":
            file_index[src]["imports"].append(label)

    # de-dup & sort
    for rec in file_index.values():
        rec["classes"] = sorted(set(x for x in rec["classes"] if x))
        rec["functions"] = sorted(set(x for x in rec["functions"] if x))
        rec["imports"] = sorted(set(x for x in rec["imports"] if x))

    return file_index

def guess_area_from_path(path: str) -> str:
    p = path.lower()
    if any(x in p for x in ["infra", "deploy", "k8s", "helm", "terraform", "ansible", "docker", ".github/workflows", "ci", "cd"]):
        return "Deployment/Infrastructure"
    if any(x in p for x in ["frontend", "ui", "client", "web", "components", "pages", "src/app", "src/components"]):
        return "Frontend Components"
    if any(x in p for x in ["backend", "server", "api", "controllers", "routes", "services"]):
        return "Backend Systems"
    if any(x in p for x in ["model", "ml", "nn", "inference", "training", "weights"]):
        return "Model Integration"
    if any(x in p for x in ["data", "dataset", "etl", "pipeline", "db", "sql", "repository", "storage"]):
        return "Data Management/Flow"
    if any(x in p for x in ["docs", "readme", "readme.md"]):
        return "Overview"
    return "Other"

def summarize_graph(graph: Dict[str, Any],
                    max_files_in_prompt: int = 200,
                    top_imports_k: int = 50) -> Dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    meta = graph.get("meta", {})

    file_index = build_file_index(nodes, edges)
    files: List[Dict[str, Any]] = []
    langs = Counter()
    imports_counter = Counter()
    area_counter = Counter()

    for _, rec in file_index.items():
        path = rec["path"]
        lang = rec.get("lang") or "unknown"
        classes = rec["classes"]
        functions = rec["functions"]
        imports = rec["imports"]

        langs[lang] += 1
        imports_counter.update(imports)
        area_counter[guess_area_from_path(path)] += 1

        files.append({
            "path": path,
            "lang": lang,
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "symbol_count": len(classes) + len(functions) + len(imports),
        })

    # prioritize by number of symbols (then path for stability)
    files.sort(key=lambda r: (r["symbol_count"], r["path"].lower()), reverse=True)
    files_included = files[:max_files_in_prompt]

    # top-level dir distribution
    top_dirs = Counter()
    for f in files:
        parts = f["path"].split("/")
        if len(parts) > 1:
            top_dirs[parts[0]] += 1

    summary = {
        "meta": {
            "owner": meta.get("owner", ""),
            "repo": meta.get("repo", ""),
            "branch": meta.get("branch", ""),
            "subpath": meta.get("subpath", ""),
            "created_at": meta.get("created_at", ""),
            "totals": meta.get("totals", {}),
            "graph_id": meta.get("graph_id", "graph"),
            "source_url": meta.get("source_url", ""),
        },
        "langs": langs.most_common(),
        "areas": area_counter.most_common(),
        "top_dirs": top_dirs.most_common(20),
        "top_imports": imports_counter.most_common(top_imports_k),
        "files": files_included,
        "files_total": len(files),
    }
    return summary

# ---------------------- Prompt builder ----------------------

XML_SPEC = """Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <sections>
    <section id="section-1">
      <title>[Section title]</title>
      <pages>
        <page_ref>page-1</page_ref>
        <page_ref>page-2</page_ref>
      </pages>
      <subsections>
        <section_ref>section-2</section_ref>
      </subsections>
    </section>
  </sections>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
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

def build_prompt(summary: Dict[str, Any], lang_text: str) -> str:
    meta = summary.get("meta", {})
    repo_id = f"{meta.get('owner','')}/{meta.get('repo','')}".strip("/")

    files_lines = []
    for f in summary["files"]:
        files_lines.append({
            "path": f["path"],
            "lang": f["lang"],
            "classes": f["classes"][:12],
            "functions": f["functions"][:12],
            "imports": f["imports"][:12],
        })

    prompt = f"""You are a senior documentation architect.

I want to create a wiki for this repository: {repo_id or '(unknown)'}
Source: {meta.get('source_url', '(unknown)')}

Determine the most logical structure for a wiki based on the repository's content.

IMPORTANT: The wiki content will be generated in {lang_text} language.

When designing the wiki structure, include pages that would benefit from visual diagrams, such as (this is not an exhaustive list, you can add or remove as per the need of the repository):
- Architecture overviews
- State machines
- Class hierarchies

Create a structured wiki with the following main sections:
- Overview (general information about the project)
- System Architecture (how the system is designed)
- Core Features (key functionality)
- Data Management/Flow
- Frontend Components
- Backend Systems
- Model Integration
- Deployment/Infrastructure
- Extensibility and Customization

Guidelines:
- Use the repository's structure, file languages, and imports to infer system boundaries and features.
- Group related pages and include cross-links (related_pages).
- Assign an importance level per page (high/medium/low).
- Include <relevant_files> for each page (use file paths from the summary).
- If some sections do not apply, omit them. Add additional sections if clearly justified.
- Prefer concise, well-structured XML that strictly follows the schema.

Repository summary (auto-generated):
- Totals: {json.dumps(summary.get('meta', {}).get('totals', {}), ensure_ascii=False)}
- Languages (by file count): {summary.get('langs', [])}
- Top-level directories: {summary.get('top_dirs', [])}
- Heuristic areas (from paths): {summary.get('areas', [])}
- Top imports: {summary.get('top_imports', [])}
- Files included (up to {len(files_lines)} of {summary.get('files_total', 0)} total):
{json.dumps(files_lines, ensure_ascii=False, indent=2)}

{XML_SPEC}

STRICT REQUIREMENTS:
- Output ONLY the <wiki_structure> XML. Do not include any markdown or commentary.
- Ensure all <section_ref> and <page_ref> refer to defined IDs.
- Ensure each <page> has exactly one <parent_section>.
- Use {lang_text} for all natural-language text fields (titles, descriptions).
"""
    return prompt

# ---------------------- QGenie generation ----------------------

def qgenie_generate_xml(
    prompt_text: str,
) -> str:
    """
    Local-only generation using AutoTokenizer + AutoModelForCausalLM.
    If the tokenizer supports a chat template, we wrap the prompt as a 'user' message.
    """
    from qgenie import ChatMessage, QGenieClient

    client = QGenieClient()
 
 
    chat_response = client.chat(
    messages=[
        ChatMessage(role="user", content=prompt_text)
    ],
    model = "qwen2.5-14b-1m"
)
 
    return chat_response.first_content
    





#-----------------------QEff model generation-------------------
def qeff_generate_xml(
    model_id_or_path: str,
    prompt_text: str,
    cache_dir: Optional[str] = None,
    max_new_tokens: int = 1200,
    temperature: float = 0.2,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    do_sample: bool = False,
    eos_string: Optional[str] = "</wiki_structure>",
) -> str:
    """
    Generate XML using QEff.
    This implementation is specific to QEff and is not meant to be a general solution.
    """
    import torch
    from transformers import AutoTokenizer
    from QEfficient import QEFFAutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        model_id_or_path,
        use_fast=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Compose input (chat template if available)
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a helpful, precise documentation architect."},
            {"role": "user", "content": prompt_text},
        ]
        model_input = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        model_input = prompt_text


    model=QEFFAutoModelForCausalLM.from_pretrained(model_id_or_path)
    print("compiling")
    import time
    t1 = time.time()
    model.compile()
    t2 = time.time()
    print("compilation done in: ", t2-t1)
    # pad token id fallback
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    inputs = tok(model_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
    }

    # Try to map the closing XML tag to a token id
    if eos_string:
        try:
            # Using tokenizer to find EOS token id for the string
            eos_ids = tok(eos_string, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
            # Some tokenizers split into multiple tokens; pass list to eos_token_id only if length==1
            if len(eos_ids) == 1:
                gen_kwargs["eos_token_id"] = eos_ids[0]
            else:
                # Fall back to substring cut after generation
                pass
        except Exception:
            pass

    
    text = model.generate(prompts =[model_input],tokenizer = tok, **gen_kwargs)

    # text = tok.decode(out[0], skip_special_tokens=True)

    # If using a chat template, the decoded text may include the prompt; trim it
    if model_input and text.startswith(model_input):
        text = text[len(model_input):]

    # Strong fallback: cut at explicit closing tag if present
    if eos_string and eos_string in text:
        text = text.split(eos_string)[0] + eos_string

    return text.strip()



# ---------------------- Output helpers ----------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def default_output_path(graph_meta: Dict[str, Any]) -> str:
    graph_id = graph_meta.get("graph_id", "graph")
    out_dir = ensure_dir(os.path.join(os.getcwd(), ".cache", "wiki_structures"))
    fname = f"{graph_meta.get('owner','owner')}__{graph_meta.get('repo','repo')}__{graph_meta.get('branch','branch')}__{graph_id}__wiki.xml"
    fname = fname.replace("/", "_")
    return os.path.join(out_dir, fname)

def validate_or_wrap_xml(text: str) -> str:
    t = text.strip()
    start = t.find("<wiki_structure")
    end = t.rfind("</wiki_structure>")
    if start != -1 and end != -1:
        return t[start:end + len("</wiki_structure>")].strip()
    # wrap if the model didn't follow instructions exactly
    return f"<wiki_structure>\n{text}\n</wiki_structure>"

# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate wiki XML from graph JSON using a local HF model (transformers).")
    p.add_argument("--graph", required=True, help="Path to graph JSON file (from .cache/graphs)")
    p.add_argument("--out", default=None, help="Output XML path (default: auto in .cache/wiki_structures)")
    p.add_argument("--lang-text", required=True, help="Target language for wiki content (e.g., 'English', 'Hindi').")
    p.add_argument("--model", required=False, help="HF model id or local model directory (for local generation).")
    p.add_argument("--cache-dir", default=None, help="Hugging Face cache directory (optional)")

    # Prompt shaping
    p.add_argument("--max-files-in-prompt", type=int, default=100, help="Max files to include in prompt summary (default 200).")
    p.add_argument("--top-imports-k", type=int, default=20, help="Top-K imports to include (default 50).")

    # Generation params
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--do-sample", action="store_true", help="Enable sampling (default off).")
    p.add_argument("--eos-string", default="</wiki_structure>", help="Stop when this substring appears (default </wiki_structure>).")

    return p.parse_args()

def main():
    args = parse_args()

    # Load and summarize graph
    graph = load_graph(args.graph)
    summary = summarize_graph(graph, max_files_in_prompt=args.max_files_in_prompt, top_imports_k=args.top_imports_k)
    prompt = build_prompt(summary, lang_text=args.lang_text)
    print(prompt)
    # Generate with qgenie
    xml_text = qgenie_generate_xml(
        prompt_text=prompt
    )

    # Generate locally
    # xml_text = qeff_generate_xml(
    #     model_id_or_path=args.model,
    #     prompt_text=prompt,
    #     cache_dir=args.cache_dir,
    #     max_new_tokens=args.max_new_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     repetition_penalty=args.repetition_penalty,
    #     do_sample=args.do_sample,
    #     eos_string=args.eos_string,
    # )

    xml_text = validate_or_wrap_xml(xml_text)

    out_path = args.out or default_output_path(summary.get("meta", {}))
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_text)
    print(f"[OK] Wiki XML saved to: {out_path}")

if __name__ == "__main__":
    main()
