import random
import yaml

def generate_random_personality():
    """
    Generate random BIG 5 personality trait values.
    
    Returns:
        dict: Dictionary containing the five personality traits with random values
    """
    return {
        "openness": round(random.random(), 2),
        "conscientiousness": round(random.random(), 2),
        "extraversion": round(random.random(), 2),
        "agreeableness": round(random.random(), 2),
        "neuroticism": round(random.random(), 2)
    }

def get_personality_description(personality):
    """
    Return a human-readable description of the personality traits.
    
    Args:
        personality (dict): Dictionary containing BIG 5 personality traits
        
    Returns:
        str: Human-readable description of the personality
    """
    descriptions = []
    
    trait_descriptions = {
        "openness": ["conventional and traditional", "curious and innovative"],
        "conscientiousness": ["spontaneous and flexible", "organized and disciplined"],
        "extraversion": ["reserved and reflective", "outgoing and energetic"],
        "agreeableness": ["analytical and detached", "empathetic and cooperative"],
        "neuroticism": ["emotionally stable and calm", "sensitive and reactive"]
    }
    
    for trait, (low, high) in trait_descriptions.items():
        value = personality.get(trait, 0.5)
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

def save_personality(personality, file_path):
    """
    Save personality traits to a YAML file.
    
    Args:
        personality (dict): Dictionary containing BIG 5 personality traits
        file_path (str): Path to save the YAML file
    """
    with open(file_path, 'w') as file:
        yaml.dump(personality, file, default_flow_style=False)

def load_personality(file_path):
    """
    Load personality traits from a YAML file.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Dictionary containing BIG 5 personality traits
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
