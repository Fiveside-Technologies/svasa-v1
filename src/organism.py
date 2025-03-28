import os
import yaml
import random
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
        
        # Generate random BIG 5 personality traits
        self.personality = self._generate_random_personality()
        
        # Save personality traits to YAML file
        self._save_personality()

    def _generate_random_personality(self):
        """Generate random BIG 5 personality trait values."""
        return {
            "openness": round(random.random(), 2),
            "conscientiousness": round(random.random(), 2),
            "extraversion": round(random.random(), 2),
            "agreeableness": round(random.random(), 2),
            "neuroticism": round(random.random(), 2)
        }
    
    def _save_personality(self):
        """Save personality traits to YAML file."""
        with open(self.personality_file, 'w') as file:
            yaml.dump(self.personality, file, default_flow_style=False)
    
    def get_personality_description(self):
        """Return a human-readable description of the organism's personality."""
        descriptions = []
        
        trait_descriptions = {
            "openness": ["conventional and traditional", "curious and innovative"],
            "conscientiousness": ["spontaneous and flexible", "organized and disciplined"],
            "extraversion": ["reserved and reflective", "outgoing and energetic"],
            "agreeableness": ["analytical and detached", "empathetic and cooperative"],
            "neuroticism": ["emotionally stable and calm", "sensitive and reactive"]
        }
        
        for trait, (low, high) in trait_descriptions.items():
            value = self.personality[trait]
            if value < 0.3:
                descriptions.append(f"Very {low}")
            elif value < 0.5:
                descriptions.append(f"Somewhat {low}")
            elif value > 0.7:
                descriptions.append(f"Very {high}")
            elif value > 0.5:
                descriptions.append(f"Somewhat {high}")
            else:
                descriptions.append(f"Balanced between {low} and {high}")
        
        return ". ".join(descriptions) + "."

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
        
        # Set up personality directory and file path
        instance.personality_dir = os.path.join(instance.org_dir, "personality")
        os.makedirs(instance.personality_dir, exist_ok=True)
        instance.personality_file = os.path.join(instance.personality_dir, "traits.yaml")
        
        # Load personality if file exists, otherwise generate random personality
        if os.path.exists(instance.personality_file):
            with open(instance.personality_file, 'r') as file:
                instance.personality = yaml.safe_load(file)
        else:
            instance.personality = instance._generate_random_personality()
            instance._save_personality()
        
        # Check if organism exists
        if not os.path.exists(instance.org_dir):
            raise ValueError(f"Organism '{organism_name}' does not exist.")
            
        return instance

    def __repr__(self):
        return f"<Organism name={self.name} dirs={{'organism': {self.org_dir}, 'obsidian': {self.obsidian_db_dir}, 'vector': {self.vector_db_dir}}}>"
