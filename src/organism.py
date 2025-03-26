import os
from utils import generate_unique_code


class Organism:
    """A class representing an organism with a unique code-based name and dedicated folders."""
    def __init__(self):
        # Determine organisms directory relative to this file's location (project root)
        organisms_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "organisms"))
        
        # Ensure the organisms directory exists
        os.makedirs(organisms_dir, exist_ok=True)
        
        # Get a list of existing organism folder names to avoid duplicates
        existing_codes = [d for d in os.listdir(organisms_dir) if os.path.isdir(os.path.join(organisms_dir, d))]
        
        # Generate a unique name for this organism
        self.name = generate_unique_code(existing_codes)
        
        # Create a new folder for the organism
        self.org_dir = os.path.join(organisms_dir, self.name)
        os.makedirs(self.org_dir, exist_ok=True)
        
        # Create subdirectories in a pythonic manner: obsidian_db and vector_db
        self.obsidian_db_dir = os.path.join(self.org_dir, "obsidian_db")
        self.vector_db_dir = os.path.join(self.org_dir, "vector_db")
        os.makedirs(self.obsidian_db_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)

    def __repr__(self):
        return f"<Organism name={self.name} dirs={{'organism': {self.org_dir}, 'obsidian': {self.obsidian_db_dir}, 'vector': {self.vector_db_dir}}}>"
