import os
import streamlit as st
import tempfile
import zipfile
import io
import re
import uuid
 
# Custom CSS for better scrolling and layout
st.markdown("""
<style>
/* Custom styling for better layout and scrolling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
 
/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
 
/* Column styling for better scrolling */
.stColumn > div {
    height: 80vh;
    overflow-y: auto;
    padding-right: 10px;
}
 
/* Custom scrollbar */
.stColumn > div::-webkit-scrollbar {
    width: 8px;
}
 
.stColumn > div::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}
 
.stColumn > div::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
}
 
.stColumn > div::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
}
 
/* Chat history styling */
.streamlit-expander {
    border: 1px solid #e1e5e9;
    border-radius: 8px;
    margin-bottom: 10px;
}
 
.streamlit-expander .streamlit-expanderHeader {
    background-color: #f8f9fa;
    border-radius: 8px 8px 0 0;
}
 
/* Documentation content styling */
h1, h2, h3 {
    color: #2c3e50;
}
 
/* Mermaid diagram container */
.mermaid {
    margin: 20px 0;
}
 
/* Page list styling */
.stButton > button {
    margin-bottom: 8px;
    text-align: left;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    transition: all 0.3s ease;
}
 
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
 
/* Primary button styling for selected page */
.stButton > button[kind="primary"] {
    background-color: #ff6b6b;
    border-color: #ff6b6b;
}
 
/* Secondary button styling for unselected pages */
.stButton > button[kind="secondary"] {
    background-color: #f8f9fa;
    border-color: #dee2e6;
    color: #495057;
}
 
/* Sticky chat input styling - Enhanced for Streamlit */
.sticky-chat-container {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    background-color: white !important;
    border-top: 2px solid #ff6b6b !important;
    padding: 1rem !important;
    z-index: 999999 !important;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.15) !important;
    backdrop-filter: blur(10px) !important;
    background-color: rgba(255, 255, 255, 0.98) !important;
}
 
/* Ensure the sticky container is above everything */
.sticky-chat-container * {
    position: relative !important;
}
 
/* Adjust main content to account for sticky chat */
.main-content-with-sticky-chat {
    padding-bottom: 160px !important;
    margin-bottom: 20px !important;
}
 
/* Chat input styling */
.sticky-chat-input {
    max-width: 800px !important;
    margin: 0 auto !important;
}
 
/* Force Streamlit elements to respect sticky positioning */
.sticky-chat-container .stTextInput > div > div > input {
    border: 2px solid #ff6b6b !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 16px !important;
}
 
.sticky-chat-container .stButton > button {
    background-color: #ff6b6b !important;
    border-color: #ff6b6b !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: bold !important;
}
 
/* Responsive adjustments */
@media (max-width: 768px) {
    .sticky-chat-container {
        padding: 0.75rem !important;
    }
    .main-content-with-sticky-chat {
        padding-bottom: 140px !important;
    }
}
 
/* Half-and-half layout adjustments */
.stColumn .sticky-chat-container {
    position: relative;
    bottom: auto;
    left: auto;
    right: auto;
    box-shadow: none;
    border-top: none;
    padding: 0;
}
 
/* JavaScript to ensure sticky positioning works */
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Force sticky positioning
    const stickyElements = document.querySelectorAll('.sticky-chat-container');
    stickyElements.forEach(function(element) {
        element.style.position = 'fixed';
        element.style.bottom = '0';
        element.style.left = '0';
        element.style.right = '0';
        element.style.zIndex = '999999';
        element.style.width = '100%';
        element.style.backgroundColor = 'white';
        element.style.borderTop = '2px solid #ff6b6b';
        element.style.padding = '1rem';
        element.style.boxShadow = '0 -4px 20px rgba(0,0,0,0.15)';
    });
});
</script>
</style>
""", unsafe_allow_html=True)
 
from build_wiki import build_sharded_wiki_from_github, get_unique_cache_dir
from build_embeddings import build_embeddings_for_repo
from generate_wiki_pages import generate_wiki_pages
from hybrid_code_chatbot import build_hybrid_code_chatbot_graph  
 
# ========== UI: Sidebar Inputs ==========
st.set_page_config(page_title="Codebase Wiki & Chatbot", layout="wide")
st.title("DocuGenie: Codebase Documentation & Q&A Chatbot")
 
with st.sidebar:
    # Only show setup form if no documentation is loaded
    if not st.session_state.get('page_files'):
        st.header("Repository & Generation Settings")
        
        # Show a small indicator when setup form is active
        if st.session_state.get('manual_reset', False):
            st.success("Ready for new documentation setup")
        github_url = st.text_input("GitHub URL", value="https://github.com/pallets/flask")
        github_token = st.text_input("GitHub Token (optional)", type="password")
        language = st.selectbox("Documentation Language", ["English", "Chinese", "Spanish", "French"], index=0)
        model_name = st.selectbox("QGenie Model", ["Pro", "Turbo"], index=0)
        
        # Performance options
        st.subheader("Performance Options")
        max_files = st.slider("Max Files to Process", min_value=10, max_value=200, value=50, 
                             help="Fewer files = faster processing. More files = more comprehensive docs.")
        use_cache = st.checkbox("Use Caching (Recommended)", value=True, 
                               help="Cache embeddings to speed up subsequent runs")
        regenerate = st.checkbox("Regenerate Documentation (ignore cache)", value=False)
        
        generate_btn = st.button("Generate Documentation", type="primary", use_container_width=True)
    else:
        # Documentation is loaded, show clean interface
        generate_btn = False
        # Use stored values from session state
        github_url = st.session_state.get('github_url', '')
        github_token = ''
        language = 'English'
        model_name = 'Pro'
        max_files = 50
        use_cache = True
        regenerate = False
        
        # Add option to start fresh
        st.markdown("---")
        if st.button("Generate New Documentation", use_container_width=True, type="secondary"):
            # Clear all session state to show setup form again
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Set a flag to prevent auto-loading after manual reset
            st.session_state.manual_reset = True
            
            # Force rerun to show the setup form
            st.rerun()
 
# ========== Helper Functions ==========
def create_github_url(github_url, file_path, line_number=None):
    """
    Create a clickable GitHub URL for a file or symbol reference
    """
    # Check if github_url is valid
    if not github_url or not github_url.startswith('https://github.com/'):
        return file_path
    
    # Extract repo info from GitHub URL
    repo_part = github_url.replace('https://github.com/', '').rstrip('/')
    if line_number:
        return f"https://github.com/{repo_part}/blob/HEAD/{file_path}#L{line_number}"
    else:
        return f"https://github.com/{repo_part}/blob/HEAD/{file_path}"
 
def display_references_with_links(references_files, references_symbols, github_url, retrieved_files=None, retrieved_symbols=None):
    """
    Display references as clickable GitHub links
    """
    if references_files or references_symbols:
        st.markdown("**📚 References:**")
        
        # Display file references
        for ref in references_files:
            github_link = create_github_url(github_url, ref)
            if github_url and github_link != ref:
                # Show as clickable link if we have a valid GitHub URL
                st.markdown(f"- File: [{ref}]({github_link})")
            else:
                # Show as plain text if no GitHub URL available
                st.markdown(f"- File: `{ref}`")
        
        # Display symbol references with better context
        if retrieved_symbols:
            # Use the full symbol data if available
            for symbol_data in retrieved_symbols:
                symbol_name = symbol_data.get("symbol_name", "")
                file_path = symbol_data.get("file_path", "")
                if file_path:
                    github_link = create_github_url(github_url, file_path)
                    if github_url and github_link != file_path:
                        # Show as clickable link if we have a valid GitHub URL
                        st.markdown(f"- Symbol: `{symbol_name}` in [{file_path}]({github_link})")
                    else:
                        # Show as plain text if no GitHub URL available
                        st.markdown(f"- Symbol: `{symbol_name}` in `{file_path}`")
                else:
                    st.markdown(f"- Symbol: `{symbol_name}`")
        else:
            # Fallback to simple symbol names
            for ref in references_symbols:
                st.markdown(f"- Symbol: `{ref}`")

# ========== Page Title Extraction ==========
def extract_page_title(markdown_content: str, filename: str) -> str:
    """Extract the main title from markdown content or use filename as fallback."""
    if not markdown_content:
        return filename.replace('.md', '').replace('_', ' ').title()
    
    # Look for the first H1 heading
    h1_match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # Look for the first H2 heading as fallback
    h2_match = re.search(r'^##\s+(.+)$', markdown_content, re.MULTILINE)
    if h2_match:
        return h2_match.group(1).strip()
    
    # Fallback to formatted filename
    return filename.replace('.md', '').replace('_', ' ').title()
 
def get_page_titles(page_files: list, page_contents: dict) -> dict:
    """Get a mapping of filenames to their extracted titles."""
    titles = {}
    for filename in page_files:
        content = page_contents.get(filename, '')
        titles[filename] = extract_page_title(content, filename)
    return titles
 
# ========== Mermaid Diagram Functions ==========
MERMAID_BLOCK_RE = re.compile(r"```mermaid\s+([\s\S]*?)```", re.IGNORECASE)
 
def extract_mermaid_blocks(markdown_text: str) -> list:
    """Extract Mermaid diagram code blocks from markdown text."""
    return [m.group(1).strip() for m in MERMAID_BLOCK_RE.finditer(markdown_text or "")]
 
def render_mermaid(mermaid_code: str, height: int = 500):
    """Render Mermaid diagrams using streamlit-mermaid library."""
    try:
        import streamlit_mermaid as stmd
        safe = (mermaid_code or "").replace("\\", "\\\\").replace("`", "\\`")
        cid = f"mermaid-{uuid.uuid4().hex}"
        stmd.st_mermaid(safe, key=cid)
    except ImportError:
        # Fallback to code block if streamlit-mermaid is not available
        st.code(mermaid_code, language="mermaid")
        st.info(" Install `streamlit-mermaid` to render diagrams interactively: `pip install streamlit-mermaid`")
    except Exception as e:
        # Fallback to code block on any error
        st.code(mermaid_code, language="mermaid")
        st.warning(f" Could not render Mermaid diagram: {e}")
 
def get_actual_file_paths_from_xml(page_filename: str) -> list:
    """Get the actual file paths used to generate a wiki page from XML metadata."""
    if not page_filename:
        return []
    
    # Get the current repository directory
    github_url = st.session_state.get('github_url')
    if not github_url:
        return []
    
    # Construct the path to the wiki XML directory
    root_cache_dir = os.path.join("..", ".cache")
    repo_name = get_unique_cache_dir(github_url).split(os.sep)[-1]
    repo_dir = os.path.abspath(os.path.join(root_cache_dir, repo_name))
    wiki_xml_dir = os.path.join(repo_dir, "wiki_xml")
    
    if not os.path.exists(wiki_xml_dir):
        return []
    
    # Map page filename to page number (e.g., "page-1.md" -> 1)
    page_match = re.match(r'page-(\d+)\.md', page_filename)
    if not page_match:
        return []
    
    page_number = int(page_match.group(1))
    
    # Look through XML files to find the page
    try:
        import xml.etree.ElementTree as ET
        
        for xml_file in os.listdir(wiki_xml_dir):
            if not xml_file.endswith('.xml'):
                continue
                
            xml_path = os.path.join(wiki_xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Find all pages in this XML file
            pages = root.findall('.//page')
            
            # Check if our page number exists in this XML file
            if page_number <= len(pages):
                page = pages[page_number - 1]
                
                # Get relevant files for this page
                relevant_files = page.find('relevant_files')
                if relevant_files is not None:
                    file_paths = []
                    for file_path_elem in relevant_files.findall('file_path'):
                        if file_path_elem.text and file_path_elem.text.strip():
                            file_paths.append(file_path_elem.text.strip())
                    return file_paths
                    
    except Exception as e:
        print(f"Error reading XML metadata: {e}")
    
    return []
 
def render_markdown_with_mermaid(markdown_content: str, page_filename: str = None):
    """Render markdown content with Mermaid diagrams and clickable references."""
    if not markdown_content:
        return
    
    # Get actual file paths from wiki XML metadata instead of parsing markdown
    references_files = get_actual_file_paths_from_xml(page_filename)
    references_symbols = []  # We'll focus on file references for now
    
    # Split content by Mermaid blocks
    parts = MERMAID_BLOCK_RE.split(markdown_content)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular markdown content
            if part.strip():
                # Remove the References section from the markdown since we'll render it separately
                cleaned_part = remove_references_section(part)
                if cleaned_part.strip():
                    st.markdown(cleaned_part, unsafe_allow_html=True)
        else:
            # Mermaid diagram code
            if part.strip():
                render_mermaid(part.strip())
    
    # Render references using the existing function
    if references_files or references_symbols:
        github_url = st.session_state.get('github_url')
        display_references_with_links(references_files, references_symbols, github_url)
 
def extract_references_from_markdown(markdown_content: str) -> tuple[list, list]:
    """Extract file and symbol references from markdown content."""
    references_files = []
    references_symbols = []
    
    if not markdown_content:
        return references_files, references_symbols
    
    lines = markdown_content.split('\n')
    in_references_section = False
    
    for line in lines:
        # Check if this line starts a References section
        if line.strip().startswith('## References') or line.strip().startswith('### References'):
            in_references_section = True
            continue
        
        # Check if we're exiting the references section (next heading or end)
        if in_references_section and line.strip().startswith('#'):
            in_references_section = False
        
        # If we're in the references section, extract file paths
        if in_references_section and line.strip():
            # Extract file paths from different formats
            file_paths = extract_file_paths_from_line(line)
            references_files.extend(file_paths)
    
    return references_files, references_symbols
 
def extract_file_paths_from_line(line: str) -> list:
    """Extract file paths from a line that might be in various formats."""
    file_paths = []
    
    # Pattern 1: Bullet points with optional description
    # Matches: - filename.py, - `filename.py`, - Description: filename.py, - Description: `filename.py`
    bullet_pattern = r'^(\s*[-*]\s*)(.*?)(?::\s*)?(`?)([^`\s]+\.(?:py|js|ts|md|txt|json|yml|yaml|xml|html|css|rst|jsx|tsx|vue|php|rb|java|cpp|c|h|hpp))(`?)(.*)$'
    bullet_match = re.match(bullet_pattern, line, re.IGNORECASE)
    
    if bullet_match:
        file_path = bullet_match.group(4)  # Actual file path
        # Clean the file path (remove leading /)
        clean_path = file_path.lstrip('/')
        file_paths.append(clean_path)
        return file_paths  # Return early to avoid duplicate matches
    
    # Pattern 2: Numbered lists
    # Matches: 1. filename.py, 1. `filename.py`, 1. Description: filename.py
    numbered_pattern = r'^(\s*\d+\.\s*)(.*?)(?::\s*)?(`?)([^`\s]+\.(?:py|js|ts|md|txt|json|yml|yaml|xml|html|css|rst|jsx|tsx|vue|php|rb|java|cpp|c|h|hpp))(`?)(.*)$'
    numbered_match = re.match(numbered_pattern, line, re.IGNORECASE)
    
    if numbered_match:
        file_path = numbered_match.group(4)
        clean_path = file_path.lstrip('/')
        file_paths.append(clean_path)
        return file_paths  # Return early to avoid duplicate matches
    
    # Pattern 3: Standalone backtick-wrapped paths (only if not already matched above)
    # Matches: `filename.py`, `path/to/file.py`
    backtick_pattern = r'(`)([^`\s]+\.(?:py|js|ts|md|txt|json|yml|yaml|xml|html|css|rst|jsx|tsx|vue|php|rb|java|cpp|c|h|hpp))(`)'
    backtick_match = re.search(backtick_pattern, line, re.IGNORECASE)
    
    if backtick_match:
        file_path = backtick_match.group(2)
        clean_path = file_path.lstrip('/')
        file_paths.append(clean_path)
        return file_paths  # Return early to avoid duplicate matches
    
    # Pattern 4: Paths starting with / (like `/flask/werkzeug/`)
    # These might be directories, but we'll include them as potential references
    slash_pattern = r'(\s*)(`?)(/[^`\s]+)(`?)(\s*.*)$'
    slash_match = re.match(slash_pattern, line)
    
    if slash_match and not any(ext in line for ext in ['.py', '.js', '.ts', '.md', '.txt']):
        path_part = slash_match.group(3).lstrip('/')
        # Only include if it looks like a reasonable path (not just a word)
        if '/' in path_part or len(path_part) > 3:
            file_paths.append(path_part)
    
    return file_paths
 
def remove_references_section(content: str) -> str:
    """Remove the References section from markdown content."""
    if not content:
        return content
    
    lines = content.split('\n')
    filtered_lines = []
    in_references_section = False
    
    for line in lines:
        # Check if this line starts a References section
        if line.strip().startswith('## References') or line.strip().startswith('### References'):
            in_references_section = True
            continue
        
        # Check if we're exiting the references section (next heading or end)
        if in_references_section and line.strip().startswith('#'):
            in_references_section = False
        
        # Skip lines in the references section
        if in_references_section:
            continue
        
        # Keep all other lines
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)
 
# ========== Pipeline Execution ==========
def run_pipeline(github_url, github_token, language, model_name, regenerate, max_files=50, use_cache=True):
    # Use standardized cache location in root .cache directory
    root_cache_dir = os.path.join("..", ".cache")  # Go up from core/ to root, then into .cache
    repo_name = get_unique_cache_dir(github_url).split(os.sep)[-1]  # Extract repo name
    repo_dir = os.path.join(root_cache_dir, repo_name)
    # Convert to absolute path to handle working directory issues
    repo_dir = os.path.abspath(repo_dir)
    os.makedirs(repo_dir, exist_ok=True)
    cache_exists = os.path.exists(os.path.join(repo_dir, "wiki_pages"))
 
    # Use cache if available and not regenerating
    if cache_exists and not regenerate:
        st.success(" Loaded documentation from cache.")
    else:
        if regenerate:
            st.info(" Regenerating documentation (ignoring cache)...")
        else:
            st.info(" Building documentation from scratch...")
            
        with st.spinner("Building sharded wiki..."):
            build_sharded_wiki_from_github(
                gh_url=github_url,
                token=github_token,
                output_root=repo_dir,
                lang_text=language,
                qgenie_model=model_name,
            )
        with st.spinner("Building code embeddings..."):
            # Use optimized embedding builder with caching
            embeddings_dir = build_embeddings_for_repo(
                github_url=github_url,
                token=github_token,
                output_root=repo_dir,
                MAX_FILES=max_files,
                force_summarize=regenerate  # Force re-summarization if regenerating
            )
        with st.spinner("Generating documentation pages..."):
            generate_wiki_pages(
                output_root=repo_dir,
                lang_text=language,
            )
        st.success(" Documentation generated and cached.")
 
    # Load documentation and chatbot (always executed)
    wiki_pages_dir = os.path.join(repo_dir, "wiki_pages")
    if os.path.exists(wiki_pages_dir):
        page_files = sorted([f for f in os.listdir(wiki_pages_dir) if f.endswith(".md")])
        page_contents = {}
        for f in page_files:
            with open(os.path.join(wiki_pages_dir, f), "r", encoding="utf-8") as fp:
                page_contents[f] = fp.read()
    else:
        page_files = []
        page_contents = {}
    
    # Build LangGraph chatbot
    chatbot_graph = build_hybrid_code_chatbot_graph(os.path.join(repo_dir, "embeddings"), model_name=model_name)
    return page_files, page_contents, chatbot_graph, repo_dir
 
# ========== Session State ==========
if "page_files" not in st.session_state:
    st.session_state.page_files = []
if "page_contents" not in st.session_state:
    st.session_state.page_contents = {}
if "chatbot_graph" not in st.session_state:
    st.session_state.chatbot_graph = None
if "repo_dir" not in st.session_state:
    st.session_state.repo_dir = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "github_url" not in st.session_state:
    st.session_state.github_url = None

# ========== Run Pipeline ==========
if generate_btn:
    st.session_state.page_files, st.session_state.page_contents, st.session_state.chatbot_graph, st.session_state.repo_dir = run_pipeline(
        github_url, github_token, language, model_name, regenerate, max_files, use_cache
    )
    st.session_state.github_url = github_url  # Store GitHub URL for reference links
    st.session_state.chat_history = []
    # Clear the manual reset flag since we're generating new documentation
    st.session_state.manual_reset = False
 
# ========== Auto-load Cached Documentation ==========
# If no documentation is loaded but cache exists for the current URL, load it automatically
# Skip auto-loading if user manually reset the application
if not st.session_state.get('page_files') and github_url and github_url.strip() and not st.session_state.get('manual_reset', False):
    # Use standardized cache location in root .cache directory
    root_cache_dir = os.path.join("..", ".cache")  # Go up from core/ to root, then into .cache
    possible_cache_dirs = [
        os.path.join(root_cache_dir, get_unique_cache_dir(github_url).split(os.sep)[-1]),  # Root .cache directory
        get_unique_cache_dir(github_url),  # Fallback to old location
    ]
    
    repo_dir = None
    for cache_dir in possible_cache_dirs:
        abs_cache_dir = os.path.abspath(cache_dir)
        wiki_pages_dir = os.path.join(abs_cache_dir, "wiki_pages")
        if os.path.exists(wiki_pages_dir) and os.listdir(wiki_pages_dir):
            # Check if it has markdown files
            md_files = [f for f in os.listdir(wiki_pages_dir) if f.endswith('.md')]
            if md_files:
                repo_dir = abs_cache_dir
                break
    
    if repo_dir:
        wiki_pages_dir = os.path.join(repo_dir, "wiki_pages")
    else:
        # Fallback to original logic
        repo_dir = os.path.abspath(get_unique_cache_dir(github_url))
        wiki_pages_dir = os.path.join(repo_dir, "wiki_pages")
    
    # Check if cache exists and load it
    if os.path.exists(wiki_pages_dir):
        try:
            # Load documentation from cache
            all_files = os.listdir(wiki_pages_dir)
            page_files = sorted([f for f in all_files if f.endswith(".md")])
            
            page_contents = {}
            for f in page_files:
                with open(os.path.join(wiki_pages_dir, f), "r", encoding="utf-8") as fp:
                    page_contents[f] = fp.read()
            
            # Load chatbot
            embeddings_dir = os.path.join(repo_dir, "embeddings")
            chatbot_graph = build_hybrid_code_chatbot_graph(embeddings_dir, model_name=model_name)
            
            # Update session state
            st.session_state.page_files = page_files
            st.session_state.page_contents = page_contents
            st.session_state.chatbot_graph = chatbot_graph
            st.session_state.repo_dir = repo_dir
            st.session_state.github_url = github_url  # Store GitHub URL for reference links
            
            # Show success message
            st.success(f" Automatically loaded {len(page_files)} documentation pages from cache.")
            
            # Force rerun to update the UI
            st.rerun()
            
        except Exception as e:
            st.warning(f" Could not load cached documentation: {e}")
            import traceback
            st.error(f"Error details: {traceback.format_exc()}")
 
# ========== Sidebar Navigation ==========
# Initialize page_titles to empty dict to prevent KeyError
page_titles = {}
 
if st.session_state.get('page_files'):
    # Extract page titles
    page_titles = get_page_titles(st.session_state.page_files, st.session_state.page_contents)
    
    # Page navigation in sidebar
    with st.sidebar:
        st.header(" Documentation Pages")
        
        # Initialize selected page in session state if not exists
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = st.session_state.get('page_files', [None])[0]
        
        # Page selection as clickable list
        st.markdown("**Select a page:**")
        
        # Create a container for the page list
        page_list_container = st.container()
        
        with page_list_container:
            page_files = st.session_state.get('page_files', [])
            for i, filename in enumerate(page_files):
                title = page_titles[filename]
                
                # Create a button for each page
                if st.button(
                    title, 
                    key=f"page_btn_{i}",
                    use_container_width=True,
                    type="primary" if filename == st.session_state.selected_page else "secondary"
                ):
                    st.session_state.selected_page = filename
                    st.rerun()
        
        selected_page = st.session_state.get('selected_page')
        
        # Download button in sidebar
        if st.button(" Download All Documentation", use_container_width=True):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                page_contents = st.session_state.get('page_contents', {})
                for fname, content in page_contents.items():
                    zf.writestr(fname, content)
            st.download_button(
                label="Download ZIP",
                data=buf.getvalue(),
                file_name="documentation.zip",
                mime="application/zip",
                use_container_width=True,
            )
 
# ========== Main Content Area ==========
if st.session_state.get('page_files'):
    # Check if user has asked a question (for half-and-half layout)
    if st.session_state.get('chat_history'):
        # Half-and-half layout when chat is active
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            # Documentation content (left half) with separate scrolling
            page_title = page_titles.get(selected_page, selected_page) if selected_page else "Documentation"
            st.header(f" {page_title}")
            
            # Create a container with fixed height for scrolling
            doc_container = st.container()
            with doc_container:
                if selected_page and st.session_state.page_contents:
                    render_markdown_with_mermaid(st.session_state.page_contents[selected_page], selected_page)
                else:
                    st.info("No documentation content available.")
        
        with col2:
            # Chat interface (right half) with separate scrolling
            st.header(" Q&A Chatbot")
            
            # Chat input section (fixed at top)
            chat_input_container = st.container()
            with chat_input_container:
                user_question = st.text_input("Ask your question here:", key="chat_input")
                if st.button("Ask", use_container_width=True):
                    if user_question.strip():
                        # Prepare state with history for multi-turn
                        state = {
                            "question": user_question,
                            "history": st.session_state.get('chat_history', []),
                            "topk_file": 5,
                            "topk_symbol": 5,
                        }
                        result = st.session_state.chatbot_graph.invoke(state)
                        answer = result["answer"]
                        references_files = [f["file_path"] for f in result.get("retrieved_files", [])]
                        references_symbols = [s["symbol_name"] for s in result.get("retrieved_symbols", [])]
                        # Save to chat history with full retrieved data for better links
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": answer,
                            "references_files": references_files,
                            "references_symbols": references_symbols,
                            "retrieved_files": result.get("retrieved_files", []),
                            "retrieved_symbols": result.get("retrieved_symbols", []),
                        })
                        st.rerun()
            
            # Chat history section with scrolling
            if st.session_state.get('chat_history'):
                st.markdown("---")
                st.subheader(" Chat History")
                
                # Create a scrollable container for chat history
                chat_container = st.container()
                with chat_container:
                    chat_history = st.session_state.get('chat_history', [])
                    for idx, entry in enumerate(chat_history):
                        with st.expander(f"Q{idx+1}: {entry['question'][:50]}..." if len(entry['question']) > 50 else f"Q{idx+1}: {entry['question']}", expanded=(idx == len(chat_history) - 1)):
                            st.markdown(f"**Question:** {entry['question']}")
                            st.markdown(f"**Answer:** {entry['answer']}")
                            # Display references with clickable GitHub links
                            # Use session state GitHub URL or fallback to current sidebar input
                            current_github_url = st.session_state.github_url or github_url
                            display_references_with_links(
                                entry["references_files"], 
                                entry["references_symbols"], 
                                current_github_url,
                                entry.get("retrieved_files"),
                                entry.get("retrieved_symbols")
                            )

    else:
            # Default layout (full width documentation + sticky chat input at bottom)
            # Add padding to account for sticky chat
            st.markdown('<div class="main-content-with-sticky-chat">', unsafe_allow_html=True)
            
            page_title = page_titles.get(selected_page, selected_page) if selected_page else "Documentation"
            st.header(f"{page_title}")
            
            # Documentation content
            st.markdown("---")
            if selected_page and st.session_state.page_contents:
                render_markdown_with_mermaid(st.session_state.page_contents[selected_page], selected_page)
            else:
                st.info("No documentation content available.")
            
            # Close the padding div
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Floating chat button that's always visible
            st.markdown("---")
            
            # Initialize chat modal state
            if "show_chat_modal" not in st.session_state:
                st.session_state.show_chat_modal = False
            
            # Use st.components.v1.html for proper JavaScript execution
            import streamlit.components.v1 as components
            
            fab_html = """
            <div id="floating-chat-fab" style="
                position: fixed !important;
                bottom: 20px !important;
                right: 20px !important;
                z-index: 999999 !important;
                width: 60px !important;
                height: 60px !important;
                border-radius: 50% !important;
                background-color: #ff6b6b !important;
                color: white !important;
                border: none !important;
                font-size: 24px !important;
                cursor: pointer !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                transition: all 0.3s ease !important;
            " onmouseover="this.style.transform='scale(1.1)'; this.style.boxShadow='0 6px 16px rgba(0,0,0,0.4)'" 
            onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.3)'"
            onclick="alert('Chat functionality coming soon! Use the chat input at the bottom of the page.')">
                
            </div>
            """
            
            components.html(fab_html, height=0)
            
            # Regular chat input at bottom (fallback)
            st.markdown("### Ask a Question")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                user_question = st.text_input("Ask your question about the codebase:", key="chat_input", placeholder="Type your question here...")
                if st.button("Ask Question", use_container_width=True, type="primary"):
                    if user_question.strip():
                        # Prepare state with history for multi-turn
                        state = {
                            "question": user_question,
                            "history": st.session_state.get('chat_history', []),
                            "topk_file": 5,
                            "topk_symbol": 5,
                        }
                        result = st.session_state.chatbot_graph.invoke(state)
                        answer = result["answer"]
                        references_files = [f["file_path"] for f in result.get("retrieved_files", [])]
                        references_symbols = [s["symbol_name"] for s in result.get("retrieved_symbols", [])]
                        # Save to chat history with full retrieved data for better links
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": answer,
                            "references_files": references_files,
                            "references_symbols": references_symbols,
                            "retrieved_files": result.get("retrieved_files", []),
                            "retrieved_symbols": result.get("retrieved_symbols", []),
                        })
                        # Force rerun to switch to half-and-half layout
                        st.rerun()
    
else:
    # Show different messages based on whether cache exists
    if github_url and github_url.strip():
        # Use standardized cache location in root .cache directory
        root_cache_dir = os.path.join("..", ".cache")  # Go up from core/ to root, then into .cache
        repo_name = get_unique_cache_dir(github_url).split(os.sep)[-1]  # Extract repo name
        repo_dir = os.path.join(root_cache_dir, repo_name)
        # Convert to absolute path to handle working directory issues
        repo_dir = os.path.abspath(repo_dir)
        wiki_pages_dir = os.path.join(repo_dir, "wiki_pages")
        
        if os.path.exists(wiki_pages_dir):
            st.info(" Documentation exists in cache. Click **Generate Documentation** to load it, or check the **Regenerate Documentation** box to create fresh documentation.")
        else:
            st.info(" Enter a GitHub URL and click **Generate Documentation** to begin.")
    else:
        st.info(" Enter a GitHub URL and click **Generate Documentation** to begin.")
