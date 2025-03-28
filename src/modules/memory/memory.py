# Main memory system integrating working, episodic, and semantic memory components

from .working import WorkingMemory
from .episodic import EpisodicMemory
from .procedural import ProceduralMemory
from openai import OpenAI
import pandas as pd
from scipy import spatial
from config import EMBEDDING_MODEL
import os

class Memory:
    """Main memory manager class that integrates different memory types"""
    
    def __init__(self, organism):
        self.working_memory = WorkingMemory(organism.chat_history_dir)
        self.episodic_memory = EpisodicMemory(organism.episodic_memory_dir)
        self.procedural_memory = ProceduralMemory(organism.memory_dir)
        self.reflections_embeddings_file = self.episodic_memory.embeddings_file
        
    def get_working_memory(self):
        return self.working_memory
        
    def get_episodic_memory(self):
        return self.episodic_memory
        
    def get_procedural_memory(self):
        return self.procedural_memory
        
    def save_to_episodic_memory(self):
        return self.episodic_memory.add_episodic_memory(self.working_memory)
        
    def get_reflections_embeddings_file(self):
        return self.reflections_embeddings_file
    
    def update_procedural_memory(self, what_worked, what_to_avoid):
        """Update procedural memory with new insights."""
        worked_text = " ".join(what_worked) if what_worked else ""
        avoid_text = " ".join(what_to_avoid) if what_to_avoid else ""
        return self.procedural_memory.update_guidelines(worked_text, avoid_text)

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns strings and relatedness scores, sorted from most to least related."""
    client = OpenAI()
    
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]
