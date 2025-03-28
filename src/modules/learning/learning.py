import os
from markdownify import markdownify as md
from modules.learning.wikipedia import get_page, get_links, get_url, get_summary, update_summary_and_links
from modules.learning.vector import embed_markdown_content, save_embeddings_to_file
import pandas as pd  # for DataFrames to store article sections and embeddings
from utils import sanitize_filename  # Import from utils.py instead of defining locally

def learn(seed_phrase: str, obsidian_db_dir: str, vector_db_dir: str, remaining_depth: int, visited=None):
    if visited is None:
        visited = set()
    if seed_phrase in visited:
        return
    visited.add(seed_phrase)
    
    # Skip if no content is available (e.g., Wikipedia page doesn't exist)
    page_content = get_page(seed_phrase)
    if not page_content:
        print(f"No content found for '{seed_phrase}', skipping...")
        return
        
    page_md = md(page_content, heading_style='ATX')
    links = get_links(seed_phrase)
    summary = get_summary(seed_phrase)
    updated_links, updated_summary = update_summary_and_links(links, summary)
    url = get_url(seed_phrase)
    print('Reading:', url)

    # Prepare the full markdown document
    full_md_content = f"Title: {seed_phrase}\n"
    full_md_content += f"URL: {url}\n"
    full_md_content += f"Summary: {md(updated_summary, heading_style='ATX')}\n"
    full_md_content += "All links: "
    if updated_links:
        full_md_content += " ".join([f"[[{link}]]" for link in updated_links]) + "\n"
    else:
        full_md_content += "\n"
    full_md_content += "---\n\n"
    full_md_content += page_md

    # Write markdown to file
    md_file_path = os.path.join(obsidian_db_dir, f"{seed_phrase}.md")
    with open(md_file_path, "w") as f:
        f.write(full_md_content)

    # Process for embeddings and save to vector database
    print(f"Generating embeddings for {seed_phrase}...")
    embeddings_df = embed_markdown_content(full_md_content)
    
    # Create article-specific CSV file
    sanitized_name = sanitize_filename(seed_phrase)
    vector_csv_path = os.path.join(vector_db_dir, f"{sanitized_name}.csv")
    save_embeddings_to_file(embeddings_df, vector_csv_path, append=False)  # Create a new file for each article

    if remaining_depth > 1 and updated_links:
        for link in updated_links:
            child_dir = os.path.join(obsidian_db_dir, seed_phrase.lower().replace(" ", "_"))
            os.makedirs(child_dir, exist_ok=True)
            learn(link, child_dir, vector_db_dir, remaining_depth - 1, visited)
        
    