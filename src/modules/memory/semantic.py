from .memory import strings_ranked_by_relatedness
from utils import num_tokens
import pandas as pd

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Build a query context with relevant records from the knowledge base."""
    strings, _ = strings_ranked_by_relatedness(query, df)
    introduction = 'If needed, use this grounded context to factually respond to the USER MESSAGE. Let me know if you do not have enough information or context to answer a question.'
    message = introduction
    
    for string in strings:
        record = f'\n\nRecord section:\n"""\n{string}\n"""'
        if num_tokens(message + record, model=model) > token_budget:
            break
        message += record
            
    return message