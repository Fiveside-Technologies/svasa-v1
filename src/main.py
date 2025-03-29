from organism import Organism
from modules.learning.learning import learn
from modules.learning.vector import merge_embedding_files
from modules.talking.talking import user_chat, ai_personality_test, ai_father_conversation
import os
from dotenv import load_dotenv

# Load environment variables from .env file at startup
load_dotenv()

def main():
    # Uncomment this section to create a new organism and train it
    """
    # Create a new organism
    org = Organism()
    print('Created organism:', org)

    # Define a seed phrase and recursion depth
    seed_phrase = "Solar System"
    remaining_depth = 3  # Levels of recursion (including the seed phrase)

    # Call the learn function with the seed phrase and organism's obsidian_db_dir
    learn(seed_phrase, org.obsidian_db_dir, org.vector_db_dir, remaining_depth)
    
    # After all learning is complete, create a consolidated embedding file
    print("\nCreating consolidated embedding file...")
    consolidated_file = os.path.join(org.vector_db_dir, "_embeddings.csv")
    merge_embedding_files(org.vector_db_dir, consolidated_file)
    """
    
    # Load existing organism for chat
    organism_name = "BLIVK"  # Change this to use a different organism
    try:
        org = Organism.load(organism_name)
        print(f"Loaded existing organism: {org}")
        print(org.get_personality_description())

        # Start chat with the organism
        user_chat(org)

        # Take the personality test
        # ai_personality_test(org)

        # Run the father conversation
        # ai_father_conversation(org)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please create the organism first or check if the name is correct.")


if __name__ == '__main__':
    main()
