import yaml
import datetime
import math
import os

# Define core emotions
CORE_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

class Emotions:
    """A class to manage and store emotional states for organisms."""
    
    def __init__(self, personality=None, file_path=None):
        """
        Initialize emotions with either a new baseline derived from personality or load from file.
        
        Args:
            personality (dict, optional): Personality traits to derive baseline emotions
            file_path (str, optional): Path to a YAML file to load emotions from
        """
        # Default emotional states
        self.current = {emotion: 0.5 for emotion in CORE_EMOTIONS}
        self.baseline = {emotion: 0.5 for emotion in CORE_EMOTIONS}
        self.last_updated = datetime.datetime.now()
        self.file_path = file_path
        
        # If a file path is provided and the file exists, load from it
        if file_path and os.path.exists(file_path):
            self.load_from_file(file_path)
        # Otherwise, if personality is provided, calculate baseline
        elif personality:
            self.baseline = calculate_baseline_from_personality(personality)
            self.current = self.baseline.copy()
            self.last_updated = datetime.datetime.now()
            # Save to file if a path is provided
            if file_path:
                self.save_to_file(file_path)
    
    def load_from_file(self, file_path=None):
        """
        Load emotional state from a YAML file and apply time-based decay.
        
        Args:
            file_path (str, optional): Path to the YAML file. If not provided, uses self.file_path
        """
        path = file_path or self.file_path
        if not path:
            raise ValueError("No file path provided for loading emotions")
        
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
            self.baseline = data.get('baseline', self.baseline)
            stored_current = data.get('current', self.current)
            
            timestamp = data.get('last_updated')
            if timestamp:
                try:
                    last_updated = datetime.datetime.fromisoformat(timestamp)
                    
                    # Calculate time elapsed and apply decay to emotions
                    now = datetime.datetime.now()
                    elapsed_seconds = (now - last_updated).total_seconds()
                    
                    # Apply emotion decay based on elapsed time
                    self.current = self.decay_emotions(stored_current, elapsed_seconds)
                    self.last_updated = now
                except (ValueError, TypeError):
                    # If timestamp is invalid, use current time and don't decay
                    self.current = stored_current
                    self.last_updated = datetime.datetime.now()
            else:
                # If no timestamp, don't apply decay
                self.current = stored_current
                self.last_updated = datetime.datetime.now()
        
        # Update the file path if it was provided
        if file_path:
            self.file_path = file_path
    
    def decay_emotions(self, emotions, elapsed_seconds):
        """
        Apply time-based decay to emotions, moving them closer to baseline.
        
        Args:
            emotions (dict): Current emotional states
            elapsed_seconds (float): Seconds elapsed since last update
            
        Returns:
            dict: Updated emotional states after decay
        """
        # Decay rate - emotions will move 50% toward baseline after this many seconds
        # Different emotions decay at different rates
        decay_rates = {
            "joy": 3600,        # 1 hour
            "sadness": 7200,    # 2 hours
            "anger": 1800,      # 30 minutes
            "fear": 3600,       # 1 hour
            "surprise": 900,    # 15 minutes
            "disgust": 3600     # 1 hour
        }
        
        decayed = {}
        for emotion in CORE_EMOTIONS:
            current_value = emotions.get(emotion, 0.5)
            base_value = self.baseline.get(emotion, 0.5)
            rate = decay_rates.get(emotion, 3600)
            
            # Calculate decay factor (0 = no decay, 1 = full decay to baseline)
            decay_factor = 1 - math.exp(-elapsed_seconds / rate)
            
            # Apply decay
            decayed_value = current_value + (base_value - current_value) * decay_factor
            decayed[emotion] = round(decayed_value, 2)
            
        return decayed
    
    def save_to_file(self, file_path=None):
        """
        Save emotional state to a YAML file.
        
        Args:
            file_path (str, optional): Path to save the YAML file. If not provided, uses self.file_path
        """
        path = file_path or self.file_path
        if not path:
            raise ValueError("No file path provided for saving emotions")
        
        # Update the timestamp to now before saving
        self.last_updated = datetime.datetime.now()
        
        data = {
            'current': self.current,
            'baseline': self.baseline,
            'last_updated': self.last_updated.isoformat()
        }
        
        with open(path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        
        # Update the file path if it was provided
        if file_path:
            self.file_path = file_path
    
    def to_dict(self):
        """
        Convert emotional state to a dictionary representation.
        
        Returns:
            dict: Dictionary with current, baseline emotions and last updated timestamp
        """
        return {
            'current': self.current,
            'baseline': self.baseline,
            'last_updated': self.last_updated.isoformat()
        }
    
    def get_current_emotions(self):
        """
        Get the current emotional state.
        
        Returns:
            dict: Current emotional values
        """
        return self.current
    
    def get_baseline_emotions(self):
        """
        Get the baseline emotional state.
        
        Returns:
            dict: Baseline emotional values
        """
        return self.baseline
    
    def describe_emotional_state(self):
        """
        Generate a human-readable description of the current emotional state.
        
        Returns:
            str: Plain English description of the current emotions
        """
        # Find emotions above threshold and sort by intensity
        significant_emotions = []
        for emotion, value in self.current.items():
            if value >= 0.6:
                intensity = "strongly" if value >= 0.75 else "moderately"
                significant_emotions.append((emotion, value, intensity))
        
        # Sort by intensity (highest first)
        significant_emotions.sort(key=lambda x: x[1], reverse=True)
        
        if not significant_emotions:
            return "Currently feeling relatively neutral, with no strong emotions."
        
        if len(significant_emotions) == 1:
            emotion, _, intensity = significant_emotions[0]
            return f"Currently feeling {intensity} {emotion}."
        
        if len(significant_emotions) == 2:
            emotion1, _, intensity1 = significant_emotions[0]
            emotion2, _, intensity2 = significant_emotions[1]
            return f"Currently feeling {intensity1} {emotion1} and {intensity2} {emotion2}."
        
        # For 3+ emotions, mention the top two and summarize others
        descriptions = []
        primary_emotions = significant_emotions[:2]
        for emotion, _, intensity in primary_emotions:
            descriptions.append(f"{intensity} {emotion}")
        
        others = len(significant_emotions) - 2
        if others == 1:
            emotion, _, _ = significant_emotions[2]
            return f"Currently feeling {' and '.join(descriptions)}, with a touch of {emotion}."
        else:
            return f"Currently feeling {' and '.join(descriptions)}, along with {others} other emotions."


def calculate_baseline_from_personality(personality):
    """
    Calculate emotional baseline values based on personality traits.
    
    Args:
        personality (dict): Dictionary containing BIG 5 personality traits
        
    Returns:
        dict: Baseline emotional states derived from personality
    """
    # Initialize all emotions with a moderate baseline (0.5)
    baseline = {emotion: 0.5 for emotion in CORE_EMOTIONS}
    
    # Extraversion influences (higher = more joy, less fear)
    extraversion = personality.get("extraversion", 0.5)
    baseline["joy"] = adjust_value(baseline["joy"], extraversion - 0.5)
    baseline["fear"] = adjust_value(baseline["fear"], 0.5 - extraversion)
    
    # Agreeableness influences (higher = less anger, more joy)
    agreeableness = personality.get("agreeableness", 0.5)
    baseline["anger"] = adjust_value(baseline["anger"], 0.5 - agreeableness)
    baseline["joy"] = adjust_value(baseline["joy"], agreeableness - 0.5)
    
    # Neuroticism influences (higher = more fear/sadness/anger, less joy)
    neuroticism = personality.get("neuroticism", 0.5)
    baseline["fear"] = adjust_value(baseline["fear"], neuroticism - 0.5)
    baseline["sadness"] = adjust_value(baseline["sadness"], neuroticism - 0.5)
    baseline["anger"] = adjust_value(baseline["anger"], neuroticism - 0.5)
    baseline["joy"] = adjust_value(baseline["joy"], 0.5 - neuroticism)
    
    # Openness influences (higher = more surprise, less disgust)
    openness = personality.get("openness", 0.5)
    baseline["surprise"] = adjust_value(baseline["surprise"], openness - 0.5)
    baseline["disgust"] = adjust_value(baseline["disgust"], 0.5 - openness)
    
    # Conscientiousness influences (higher = less surprise)
    conscientiousness = personality.get("conscientiousness", 0.5)
    baseline["surprise"] = adjust_value(baseline["surprise"], 0.5 - conscientiousness)
    
    # Round all values for readability
    return {k: round(v, 2) for k, v in baseline.items()}

def adjust_value(value, adjustment):
    """Adjust a value by a given amount, keeping it within 0-1 range"""
    return max(0.0, min(1.0, value + (adjustment * 0.5)))
