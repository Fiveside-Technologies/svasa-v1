import os
import yaml
import datetime
from utils import generate_unique_code
from modules.personality.emotions import Emotions
from modules.personality.personality import (
    generate_random_personality,
    get_personality_description,
    save_personality,
    load_personality
)


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
        
        # Create knowledge subdirectories in a pythonic manner: obsidian_db and vector_db
        self.obsidian_db_dir = os.path.join(self.org_dir, "knowledge", "obsidian_db")
        self.vector_db_dir = os.path.join(self.org_dir, "knowledge", "vector_db")
        os.makedirs(self.obsidian_db_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)

        # Create memory subdirectories
        self.memory_dir = os.path.join(self.org_dir, "memory")
        self.chat_history_dir = os.path.join(self.memory_dir, "chat_history")
        self.episodic_memory_dir = os.path.join(self.memory_dir, "reflections")
        os.makedirs(self.chat_history_dir, exist_ok=True)
        os.makedirs(self.episodic_memory_dir, exist_ok=True)
        
        # Create personality directory and generate personality traits
        self.personality_dir = os.path.join(self.org_dir, "personality")
        os.makedirs(self.personality_dir, exist_ok=True)
        self.personality_file = os.path.join(self.personality_dir, "traits.yaml")
        self.emotions_file = os.path.join(self.personality_dir, "emotions.yaml")
        self.personality_types_file = os.path.join(self.personality_dir, "types.yaml")
        
        # Generate random BIG 5 personality traits
        self.personality = generate_random_personality()
        
        # Save personality traits to YAML file
        save_personality(self.personality, self.personality_file)
        
        # Initialize emotions
        self.emotions = Emotions(personality=self.personality, file_path=self.emotions_file)
    
    def get_personality_description(self):
        """Return a human-readable description of the organism's personality."""
        return get_personality_description(self.personality)
    
    def get_current_emotions(self):
        """Get the current emotional state of the organism."""
        return self.emotions.get_current_emotions()
    
    def get_baseline_emotions(self):
        """Get the baseline emotional state of the organism."""
        return self.emotions.get_baseline_emotions()
    
    def get_emotional_state_description(self):
        """Get a human-readable description of the organism's current emotional state."""
        return self.emotions.describe_emotional_state()

    @classmethod
    def load(cls, organism_name):
        """Load an existing organism by name."""
        # Create an instance without generating a new name
        instance = cls.__new__(cls)
        
        # Set the name to the provided name
        instance.name = organism_name
        
        # Determine organisms directory relative to this file's location (project root)
        organisms_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "organisms"))
        
        # Set up the directory paths
        instance.org_dir = os.path.join(organisms_dir, organism_name)
        instance.obsidian_db_dir = os.path.join(instance.org_dir, "knowledge", "obsidian_db")
        instance.vector_db_dir = os.path.join(instance.org_dir, "knowledge", "vector_db")
        instance.memory_dir = os.path.join(instance.org_dir, "memory")
        instance.chat_history_dir = os.path.join(instance.memory_dir, "chat_history")
        instance.episodic_memory_dir = os.path.join(instance.memory_dir, "reflections")
        
        # Create directories if they don't exist (ensures backward compatibility)
        os.makedirs(instance.obsidian_db_dir, exist_ok=True)
        os.makedirs(instance.vector_db_dir, exist_ok=True)
        os.makedirs(instance.chat_history_dir, exist_ok=True)
        os.makedirs(instance.episodic_memory_dir, exist_ok=True)
        
        # Set up personality directory and file paths
        instance.personality_dir = os.path.join(instance.org_dir, "personality")
        os.makedirs(instance.personality_dir, exist_ok=True)
        instance.personality_file = os.path.join(instance.personality_dir, "traits.yaml")
        instance.emotions_file = os.path.join(instance.personality_dir, "emotions.yaml")
        instance.personality_types_file = os.path.join(instance.personality_dir, "types.yaml")
        # Load personality if file exists, otherwise generate random personality
        if os.path.exists(instance.personality_file):
            instance.personality = load_personality(instance.personality_file)
        else:
            instance.personality = generate_random_personality()
            save_personality(instance.personality, instance.personality_file)
        
        # Initialize emotions - will load from file if it exists or create new if it doesn't
        instance.emotions = Emotions(
            personality=instance.personality,
            file_path=instance.emotions_file
        )
        
        # Check if organism exists
        if not os.path.exists(instance.org_dir):
            raise ValueError(f"Organism '{organism_name}' does not exist.")
            
        return instance

    def __repr__(self):
        return f"<Organism name={self.name} dirs={{'organism': {self.org_dir}, 'obsidian': {self.obsidian_db_dir}, 'vector': {self.vector_db_dir}}}>"
