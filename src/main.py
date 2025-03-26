from organism import Organism
from modules.learning.learning import learn
from modules.learning.vector import merge_embedding_files
import os

def main():
    # Create a new organism to test
    org = Organism()
    print('Created organism:', org)

    # Define a seed phrase and recursion depth
    seed_phrase = "Solar System"
    remaining_depth = 3  # Levels of recursion (including the seed phrase)

    # Call the learn function with the seed phrase and organism's obsidian_db_dir
    learn(seed_phrase, org.obsidian_db_dir, org.vector_db_dir, remaining_depth)
    
    # After all learning is complete, create a consolidated embedding file (optional)
    print("\nCreating consolidated embedding file...")
    consolidated_file = os.path.join(org.vector_db_dir, "all_embeddings.csv")
    merge_embedding_files(org.vector_db_dir, consolidated_file)


if __name__ == '__main__':
    main()
