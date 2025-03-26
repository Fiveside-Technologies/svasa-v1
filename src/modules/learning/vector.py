from openai import OpenAI  # for generating embeddings
import os  # for environment variables
import pandas as pd  # for DataFrames to store article sections and embeddings 
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens

# Sections to ignore in the markdown content
SECTIONS_TO_IGNORE = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
]

GPT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1600  # Maximum tokens per chunk for splitting
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
BATCH_SIZE = 1000  # Max batch size for embedding requests

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def halved_by_delimiter(string: str, delimiter: str = "\n") -> list:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        best_index = 0
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
                best_index = i
        left = delimiter.join(chunks[:best_index+1])
        right = delimiter.join(chunks[best_index+1:])
        return [left, right]

def truncated_string(string: str, model: str, max_tokens: int, print_warning: bool = True) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    if len(encoded_string) <= max_tokens:
        return string
    truncated = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated

def split_text_into_chunks(text: str, context: str, max_tokens: int = MAX_TOKENS, model: str = GPT_MODEL, max_recursion: int = 5) -> list:
    """
    Split a text section into multiple chunks if it exceeds max_tokens.
    Returns a list of (context, text_chunk) tuples.
    """
    # Check if the current text is within token limits
    combined = context + "\n\n" + text
    if num_tokens(combined) <= max_tokens:
        return [(context, text)]
    
    # If recursion limit reached, just truncate
    if max_recursion == 0:
        truncated = truncated_string(text, model=model, max_tokens=max_tokens-num_tokens(context)-2)
        return [(context, truncated)]
    
    # Try to split at paragraph boundaries first, then sentences
    for delimiter in ["\n\n", "\n", ". "]:
        left, right = halved_by_delimiter(text, delimiter=delimiter)
        if left and right:  # Both parts have content
            # Recursively split both halves
            result = []
            for half in [left, right]:
                result.extend(split_text_into_chunks(
                    half, 
                    context, 
                    max_tokens=max_tokens,
                    model=model,
                    max_recursion=max_recursion-1
                ))
            return result
    
    # If no good split found, truncate
    truncated = truncated_string(text, model=model, max_tokens=max_tokens-num_tokens(context)-2)
    return [(context, truncated)]

def split_markdown_into_chunks(page: str) -> list[tuple[str, str]]:
    """
    Splits a markdown page into chunks for vector embeddings.
    Each chunk is a tuple (context, text) where:
      - context: a string of the heading hierarchy with level indicators, separated by newlines
      - text: the content under those headings
    
    The function handles:
      - Extracting page title from metadata
      - Skipping sections listed in SECTIONS_TO_IGNORE
      - Structuring context with level indicators (# for H1, ## for H2, etc.)
      - Adding proper spacing for better embedding
      - Splitting long text sections to stay within token limits
    """
    raw_chunks = []
    
    # First, extract the page title from metadata and split at the "---" separator
    lines = page.splitlines()
    page_title = None
    content_start = 0
    
    # Find the page title and the content divider ("---")
    for i, line in enumerate(lines):
        if line.startswith("Title:"):
            page_title = line[6:].strip()
        elif line.strip() == "---":
            content_start = i + 1
            break
    
    if not page_title:
        # If no title found in metadata, use a placeholder
        page_title = "Untitled Document"
    
    # Initialize heading tracking (level -> heading text)
    current_headings = {}
    max_level = 0
    current_content = ""
    skip_section = False

    # Process content after the "---" separator
    for line in lines[content_start:]:
        # Check if line is a heading
        heading_match = re.match(r'^(#{1,6})\s+(.*)$', line)
        
        if heading_match:
            # If we have accumulated content, save it with the current context (unless in a skipped section)
            if current_content.strip() and not skip_section:
                # Build the context string with page title and all current headings
                context_parts = [f"# {page_title}"]
                for level in range(1, max_level + 1):
                    if level in current_headings:
                        prefix = "#" * (level + 1)  # +1 because page title is H1
                        context_parts.append(f"{prefix} {current_headings[level]}")
                
                context = "\n\n".join(context_parts)  # Double newline between context items
                raw_chunks.append((context, current_content.strip()))
            
            # Reset content accumulation
            current_content = ""
            
            # Update headings based on the new heading we found
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            
            # Check if this section should be skipped
            skip_section = heading_text in SECTIONS_TO_IGNORE
            
            # Clear any deeper level headings
            for l in list(current_headings.keys()):
                if l >= level:
                    del current_headings[l]
            
            # Set the new heading at this level
            current_headings[level] = heading_text
            max_level = max(max_level, level)
        elif not skip_section:
            # Add the line to the current content
            current_content += line + "\n"
    
    # Add any remaining content as a final chunk (unless in a skipped section)
    if current_content.strip() and not skip_section:
        # Build the context string with page title and all current headings
        context_parts = [f"# {page_title}"]
        for level in range(1, max_level + 1):
            if level in current_headings:
                prefix = "#" * (level + 1)
                context_parts.append(f"{prefix} {current_headings[level]}")
        
        context = "\n\n".join(context_parts)
        raw_chunks.append((context, current_content.strip()))
    
    # Now split any chunks that are too large
    final_chunks = []
    for context, text in raw_chunks:
        final_chunks.extend(split_text_into_chunks(text, context))
    
    return final_chunks

def embed_markdown_content(page: str, page_title: str = None, api_key: str = None) -> pd.DataFrame:
    """
    Process a markdown page into chunks, get embeddings, and return a DataFrame.
    
    Args:
        page: The markdown content as a string
        page_title: Optional title to include in the dataframe for reference
        api_key: OpenAI API key (defaults to environment variable)
        
    Returns:
        DataFrame with columns: 'title', 'text', 'embedding'
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    
    # Split the markdown into chunks
    chunks = split_markdown_into_chunks(page)
    
    # Format each chunk as a combined string with newlines between context and text
    formatted_chunks = []
    for context, text in chunks:
        formatted_text = f"{context}\n\n{text}"
        formatted_chunks.append(formatted_text)
    
    # Get embeddings in batches
    all_embeddings = []
    print(f"Processing {len(formatted_chunks)} chunks for embedding...")
    
    for batch_start in range(0, len(formatted_chunks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(formatted_chunks))
        batch = formatted_chunks[batch_start:batch_end]
        print(f"  Embedding batch {batch_start} to {batch_end-1}")
        
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # In case of error, add None for embeddings to maintain alignment
            all_embeddings.extend([None] * len(batch))
    
    # Create DataFrame
    titles = [page_title] * len(formatted_chunks) if page_title else [""] * len(formatted_chunks)
    df = pd.DataFrame({
        "title": titles,
        "text": formatted_chunks,
        "embedding": all_embeddings
    })
    
    return df

def save_embeddings_to_file(df: pd.DataFrame, file_path: str, append: bool = True) -> None:
    """
    Save or append embeddings DataFrame to a CSV file.
    
    Args:
        df: DataFrame with embeddings to save
        file_path: Path to save the CSV file
        append: If True and file exists, append to it; otherwise overwrite
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if append and os.path.exists(file_path):
        # Load existing data and append
        try:
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(file_path, index=False)
            print(f"Appended {len(df)} rows to existing file {file_path}")
        except Exception as e:
            print(f"Error appending to existing file, creating new file: {e}")
            df.to_csv(file_path, index=False)
    else:
        # Create new file
        df.to_csv(file_path, index=False)
        print(f"Saved {len(df)} rows to new file {file_path}")

# Add new function to merge multiple CSV files
def merge_embedding_files(directory: str, output_file: str = None, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Merge multiple embedding CSV files into a single DataFrame or file.
    
    Args:
        directory: Directory containing the CSV files to merge
        output_file: Optional path to save the merged DataFrame (if None, only returns the DataFrame)
        pattern: File pattern to match (default: "*.csv")
        
    Returns:
        DataFrame containing all merged embeddings
    """
    import glob
    
    # Find all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(directory, pattern))
    if not csv_files:
        print(f"No CSV files found in {directory} matching pattern {pattern}")
        return pd.DataFrame()
    
    print(f"Merging {len(csv_files)} CSV files...")
    
    # Read and combine all files
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            article_name = os.path.splitext(os.path.basename(file))[0]
            if 'title' not in df.columns or df['title'].isna().all():
                df['title'] = article_name
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate if needed
    pre_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['text'])
    post_dedup = len(merged_df)
    if pre_dedup > post_dedup:
        print(f"Removed {pre_dedup - post_dedup} duplicate entries")
    
    # Save if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Saved merged embeddings to {output_file}")
    
    return merged_df

if __name__ == "__main__":
    # Directly access the file using a relative path from this module.
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../organisms/AXYLG/obsidian_db/solar_system/Quaoar.md"))
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            page = f.read()
        chunks = split_markdown_into_chunks(page)
        print(f"Found {len(chunks)} chunks in the document.\n")
        for i, (context, text) in enumerate(chunks, start=1):
            print(f"--- Chunk {i}/{len(chunks)} ---")
            token_count = num_tokens(context + "\n\n" + text)
            print(f"[{token_count} tokens]")
            print(context+text)
            print("\n" + "-"*80 + "\n")
    except Exception as e:
        print(f"Error reading file: {e}")

