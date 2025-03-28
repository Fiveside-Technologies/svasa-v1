import os
import json
from openai import OpenAI
from config import GPT_MODEL
from modules.learning.vector import embed_reflection, save_embeddings_to_file
import pandas as pd
import ast

class EpisodicMemory:
    """Manages episodic memories - reflections on past conversations."""
    
    def __init__(self, episodic_dir):
        """Initialize the episodic memory manager."""
        # The episodic_dir parameter is actually the reflections directory on disk
        self.reflections_dir = episodic_dir
        os.makedirs(self.reflections_dir, exist_ok=True)
        
        # Path to store embeddings
        self.embeddings_file = os.path.join(self.reflections_dir, "_embeddings.csv")
    
    def format_conversation(self, messages):
        """
        Format the conversation messages into a readable string, skipping the system message.
        
        Args:
            messages: List of message dictionaries from working memory
            
        Returns:
            String representation of the conversation
        """
        conversation = []
        
        # Skip the system message (index 0)
        for message in messages[1:]:
            role = message["role"].upper()
            content = message["content"]
            conversation.append(f"{role}: {content}")
        
        # Join with newlines
        return "\n".join(conversation)
    
    def create_reflection_prompt(self, formatted_conversation):
        """
        Create the prompt for generating a reflection on the conversation.
        
        Args:
            formatted_conversation: String representation of the conversation
            
        Returns:
            Prompt string for the LLM
        """
        return f"""
Analyze this conversation and create a memory reflection in JSON format with these fields:
1. context_tags: 2-4 specific keywords identifying the conversation topic (e.g., "transformer_architecture")
2. conversation_summary: One sentence describing what the conversation accomplished
3. what_worked: Most effective approach used in this conversation
4. what_to_avoid: Most important pitfall to avoid in similar discussions

Be extremely concise - each field should be one clear, actionable sentence.
Use "N/A" for any field where you don't have enough information.

Output valid JSON only:
{{
    "context_tags": [string, ...],
    "conversation_summary": string,
    "what_worked": string,
    "what_to_avoid": string
}}

Here is the conversation:
{formatted_conversation}
"""
    
    def create_reflection(self, messages):
        """
        Generate a reflection on the conversation using the LLM.
        
        Args:
            messages: List of message dictionaries from working memory
            
        Returns:
            Dictionary containing the reflection
        """
        client = OpenAI()
        
        # Format the conversation
        formatted_conversation = self.format_conversation(messages)
        
        # Skip if conversation is too short (just system message or empty)
        if not formatted_conversation:
            return None
            
        # Create the reflection prompt
        reflection_prompt = self.create_reflection_prompt(formatted_conversation)
        
        # Call the API to generate the reflection
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": reflection_prompt}],
            temperature=0.2
        )
        
        # Parse the response as JSON
        try:
            reflection_text = response.choices[0].message.content.strip()
            
            # Clean up any markdown formatting if present
            if "```json" in reflection_text:
                reflection_text = reflection_text.split("```json")[1].split("```")[0].strip()
            elif "```" in reflection_text:
                reflection_text = reflection_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(reflection_text)
        except Exception as e:
            print(f"Error parsing reflection: {e}")
            # Return a default reflection if parsing fails
            return {
                "context_tags": [],
                "conversation_summary": "N/A",
                "what_worked": "N/A",
                "what_to_avoid": "N/A"
            }
    
    def add_episodic_memory(self, working_memory):
        """
        Create a reflection on a conversation and store it as an episodic memory.
        
        Args:
            working_memory: WorkingMemory instance containing the conversation
            
        Returns:
            Path to the saved memory file
        """
        # Get all messages from working memory
        messages = working_memory.get_messages()
        
        # Skip if no messages or just system message
        if not messages or len(messages) < 2:
            print("Not enough messages to create a reflection")
            return None
        
        # Format the conversation
        formatted_conversation = self.format_conversation(messages)
        
        # Create a reflection
        reflection = self.create_reflection(messages)
        
        if not reflection:
            print("Could not generate a reflection")
            return None
        
        # Create the memory entry
        memory_entry = {
            "conversation_start_time": working_memory.conversation_start_time,
            "conversation": formatted_conversation,
            "context_tags": reflection["context_tags"],
            "conversation_summary": reflection["conversation_summary"],
            "what_worked": reflection["what_worked"],
            "what_to_avoid": reflection["what_to_avoid"]
        }
        
        # Create a filename using the conversation start time for easy reference
        filename = f"{working_memory.conversation_start_time}.json"
        filepath = os.path.join(self.reflections_dir, filename)
        
        # Save as JSON
        with open(filepath, "w") as f:
            json.dump(memory_entry, f, indent=2)
        
        print(f"Reflection saved to: {filepath}")
        
        # Generate embeddings and save to CSV
        try:
            # Create embeddings for the reflection (may produce multiple chunks for long conversations)
            reflection_df = embed_reflection(filepath)
            
            # Save to the embeddings file (appending if it exists)
            save_embeddings_to_file(reflection_df, self.embeddings_file)
        except Exception as e:
            print(f"Error saving embeddings: {e}")
        
        return filepath

    def get_relevant_reflection(self, query: str, top_n: int = 1):
        """
        Find the most relevant reflection for a query using vector similarity.
        
        Args:
            query: The user's query
            top_n: Number of top reflections to return
            
        Returns:
            List of reflection dictionaries with conversation and insights
        """
        # Check if embeddings file exists
        if not os.path.exists(self.embeddings_file):
            return []
            
        try:
            # Load embeddings
            df = pd.read_csv(self.embeddings_file)
            df['embedding'] = df['embedding'].apply(ast.literal_eval)
            
            # Get most relevant memories
            from .memory import strings_ranked_by_relatedness
            strings, _ = strings_ranked_by_relatedness(query, df, top_n=top_n)
            
            results = []
            for memory_text in strings:
                # Parse memory text to extract key information
                reflection = {}
                
                if "TAGS:" in memory_text:
                    reflection["tags"] = memory_text.split("TAGS:")[1].split("\n")[0].strip()
                
                if "SUMMARY:" in memory_text:
                    reflection["summary"] = memory_text.split("SUMMARY:")[1].split("\n")[0].strip()
                
                if "WHAT WORKED:" in memory_text:
                    reflection["what_worked"] = memory_text.split("WHAT WORKED:")[1].split("\n")[0].strip()
                
                if "WHAT TO AVOID:" in memory_text:
                    reflection["what_to_avoid"] = memory_text.split("WHAT TO AVOID:")[1].split("\n")[0].strip()
                
                if "FULL CONVERSATION" in memory_text:
                    reflection["conversation"] = memory_text.split("FULL CONVERSATION")[1].strip()
                else:
                    reflection["conversation"] = memory_text
                
                results.append(reflection)
            
            return results
            
        except Exception as e:
            print(f"Error retrieving episodic memories: {e}")
            return []
