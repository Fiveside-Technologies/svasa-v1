import ast
from openai import OpenAI
import pandas as pd
from config import GPT_MODEL
from modules.memory.memory import Memory
from modules.memory.semantic import query_message
import os

# Track conversation history and insights over time
conversations = []
what_worked = set()
what_to_avoid = set()

def prompt_coordinator(
    query: str,
    knowledge_df: pd.DataFrame,
    memory,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500
) -> list:
    """
    Coordinate different memory sources to create a comprehensive prompt.
    Combines semantic knowledge with episodic reflections.
    
    Args:
        query: User's query
        knowledge_df: DataFrame with knowledge embeddings
        memory: Memory instance
        model: Language model to use
        token_budget: Token limit for context
        
    Returns:
        List of message dictionaries for the LLM
    """
    global conversations, what_worked, what_to_avoid

    # Get working memory
    working_memory = memory.get_working_memory()
    
    # Get conversation history excluding the most recent user message
    messages = working_memory.get_messages()[:-1].copy()
    
    # Step 1: Get semantic knowledge context 
    knowledge_context = query_message(
        query, 
        knowledge_df, 
        model=model, 
        token_budget=token_budget // 2  # Reserve half for episodic
    )
    
    # Step 2: Get episodic reflections
    episodic_memory = memory.get_episodic_memory()
    reflections = episodic_memory.get_relevant_reflection(query, top_n=1)
    
    # Step 3: Get procedural guidelines
    procedural_memory = memory.get_procedural_memory().get_guidelines()
    
    # Step 4: Create the coordinated prompt
    if reflections:
        reflection = reflections[0]
        
        # Get the current conversation
        current_conversation = reflection.get("conversation", "")
        
        # Update memory stores, excluding current conversation from history
        if current_conversation and current_conversation not in conversations:
            conversations.append(current_conversation)
        
        # Split insights by periods and update global sets
        if reflection.get("what_worked") and reflection["what_worked"] != "N/A":
            what_worked.update(reflection["what_worked"].split('. '))
        
        if reflection.get("what_to_avoid") and reflection["what_to_avoid"] != "N/A":
            what_to_avoid.update(reflection["what_to_avoid"].split('. '))
        
        # Get previous conversations excluding the current one
        previous_convos = [conv for conv in conversations[-4:] if conv != current_conversation][-3:]
        
        # Create episodic context with procedural memory
        context = f"""
You recall similar conversations with the user, here are the details:

Current Conversation Match: {current_conversation}
Previous Conversations: {' | '.join(previous_convos)}
What has worked well: {' '.join(what_worked)}
What to avoid: {' '.join(what_to_avoid)}

Additionally, here are guidelines for interactions with the current user:
{procedural_memory}

Use these memories, as well as previous messages in this conversation, as context for your response to the user."""

        user_message = f"\n\nUSER MESSAGE: {query}"
        final_query = context + "\n\n" + knowledge_context + user_message
    else:
        # If no reflections, just use knowledge context and procedural memory
        context = f"""
Additionally, here are guidelines for interactions with the current user:
{procedural_memory}

Use these guidelines, as well as previous messages in this conversation, as context for your response to the user."""

        user_message = f"\n\nUSER MESSAGE: {query}"
        final_query = context + "\n\n" + knowledge_context + user_message
    
    # Add the enhanced query as the last message
    messages.append({"role": "user", "content": final_query})
    
    # DEBUG: Print the full prompt
    print("\n" + "="*25 + " FULL PROMPT " + "="*25)
    print(final_query)
    print("="*60 + "\n")
    
    return messages

def ask(
    query: str,
    df: pd.DataFrame,
    memory,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500
) -> str:
    """Process a user query using RAG and return AI response."""
    client = OpenAI()
    working_memory = memory.get_working_memory()
    
    # Initialize with system message if first interaction
    if not working_memory.get_messages():
        working_memory.initialize_with_system_message(
            "You converse with the human."
        )
    
    # Store original query in working memory
    working_memory.add_user_message(query)
    
    # Use prompt coordinator to create the comprehensive prompt
    messages = prompt_coordinator(
        query,
        df,
        memory,
        model=model,
        token_budget=token_budget
    )
    
    # Call the API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5
    )
    
    response_message = response.choices[0].message.content
    
    # Store response in working memory
    working_memory.add_assistant_message(response_message)
    
    # Update procedural memory with accumulated insights
    memory.update_procedural_memory(what_worked, what_to_avoid)
    
    return response_message

def user_chat(organism):
    """Run interactive chat session with the organism's knowledge base."""
    memory = Memory(organism)
    
    # Load knowledge embeddings
    knowledge_path = os.path.join(organism.vector_db_dir, "_embeddings.csv")
    knowledge_df = pd.DataFrame(columns=["text", "embedding"])
    
    if os.path.exists(knowledge_path):
        knowledge_df = pd.read_csv(knowledge_path)
        knowledge_df['embedding'] = knowledge_df['embedding'].apply(ast.literal_eval)
    else:
        print(f"Warning: Knowledge embeddings not found at {knowledge_path}")
    
    print("Enter your questions below (type 'exit' to quit):")
    try:
        while True:
            user_query = input("USER: ").strip()
            if user_query.lower() in ("exit", "quit", "exit()", "quit()"):
                print("Exiting chat.")
                break
            answer = ask(user_query, knowledge_df, memory)
            print(f"{organism.name}: {answer}")
    finally:
        # Save conversation history and reflection
        chat_path = memory.get_working_memory().save_conversation()
        print(f"Chat history saved to: {chat_path}")
        
        reflection_path = memory.save_to_episodic_memory()
        if reflection_path:
            print(f"Reflection saved to: {reflection_path}")
        
        # Final update to procedural memory before exiting
        if what_worked or what_to_avoid:
            print("Updating procedural memory with conversation insights...")
            memory.update_procedural_memory(what_worked, what_to_avoid)

