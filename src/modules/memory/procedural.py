"""
Procedural memory module for storing and retrieving procedural knowledge.
This implementation focuses on maintaining guidelines for interactions.
"""
import os
from openai import OpenAI
from config import GPT_MODEL

class ProceduralMemory:
    """Manages procedural memories - guidelines for how to interact."""
    
    def __init__(self, memory_dir):
        # Path to the procedural memory file in the organism's memory directory
        self.memory_file = os.path.join(memory_dir, "procedural_memory.txt")
        
        # Create the file with example content if it doesn't exist
        if not os.path.exists(self.memory_file):
            # Get the example file from the module directory
            module_dir = os.path.dirname(os.path.abspath(__file__))
            example_file = os.path.join(module_dir, "example_procedural_memory.txt")
            
            if os.path.exists(example_file):
                # Copy content from example file
                with open(example_file, "r") as source, open(self.memory_file, "w") as target:
                    target.write(source.read())
            else:
                # Fallback if example file not found
                print(f"Warning: Example procedural memory file not found at {example_file}")
                with open(self.memory_file, "w") as f:
                    f.write("1. Maintain a helpful and informative tone - Creates a positive user experience.")
    
    def get_guidelines(self):
        """Retrieve the current procedural guidelines."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return f.read()
        return ""
    
    def update_guidelines(self, what_worked, what_to_avoid):
        """Update the procedural guidelines based on new feedback."""
        client = OpenAI()
        
        # Load existing guidelines
        current_takeaways = self.get_guidelines()
        
        # Prepare the update prompt
        procedural_prompt = f"""You are maintaining a continuously updated list of the most important procedural behavior instructions for an AI assistant. Your task is to refine and improve a list of key takeaways based on new conversation feedback while maintaining the most valuable existing insights.

CURRENT TAKEAWAYS:
{current_takeaways}

NEW FEEDBACK:
What Worked Well:
{what_worked}

What To Avoid:
{what_to_avoid}

Please generate an updated list of up to 10 key takeaways that combines:
1. The most valuable insights from the current takeaways
2. New learnings from the recent feedback
3. Any synthesized insights combining multiple learnings

Requirements for each takeaway:
- Must be specific and actionable
- Should address a distinct aspect of behavior
- Include a clear rationale
- Written in imperative form (e.g., "Maintain conversation context by...")

Format each takeaway as:
[#]. [Instruction] - [Brief rationale]

The final list should:
- Be ordered by importance/impact
- Cover a diverse range of interaction aspects
- Focus on concrete behaviors rather than abstract principles
- Preserve particularly valuable existing takeaways
- Incorporate new insights when they provide meaningful improvements

Return up to but no more than 10 takeaways, replacing or combining existing ones as needed to maintain the most effective set of guidelines.
Return only the list, no preamble or explanation.
"""
        
        # Generate updated guidelines
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": procedural_prompt}],
            temperature=0.2
        )

        updated_guidelines = response.choices[0].message.content
        
        # Save the updated guidelines
        with open(self.memory_file, "w") as f:
            f.write(updated_guidelines)
            
        return updated_guidelines
