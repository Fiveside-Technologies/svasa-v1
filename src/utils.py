import random
import string
import re  # for sanitizing filenames
import tiktoken
from config import GPT_MODEL

def generate_unique_code(existing_codes):
    """Generate a five-character alphanumeric code that starts with a letter and is not in the provided list of existing codes.
    Args:
        existing_codes (list): A list of codes to avoid duplicates.

    Returns:
        str: A unique five-character code.
    """
    while True:
        # First character is a random uppercase letter
        first_char = random.choice(string.ascii_uppercase)
        # Remaining four characters can be uppercase letters or digits
        remaining = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        code = first_char + remaining
        if code not in existing_codes:
            return code

def sanitize_filename(filename: str) -> str:
    """Convert a string to a valid filename by removing invalid characters.
    
    Args:
        filename (str): The string to convert to a valid filename
        
    Returns:
        str: A sanitized filename with invalid characters removed and spaces replaced with underscores
    """
    # Replace spaces with underscores and remove invalid filename characters
    sanitized = re.sub(r'[\\/*?:"<>|]', '', filename)
    sanitized = sanitized.replace(' ', '_')
    return sanitized.lower()

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))