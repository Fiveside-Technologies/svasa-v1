import ast
from openai import OpenAI
import pandas as pd
from config import GPT_MODEL
from modules.memory.memory import Memory
from modules.memory.semantic import query_message
import os
from modules.personality.sentiment_analysis import analyze_user_message

# Track conversation history and insights over time
conversations = []
what_worked = set()
what_to_avoid = set()

def prompt_coordinator(
    query: str,
    knowledge_df: pd.DataFrame,
    memory,
    organism=None,  # Add organism parameter
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
        organism: The organism instance (optional)
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
    
    # Step 4: Get emotional state if available
    emotional_state = ""
    if organism and hasattr(organism, 'emotions'):
        emotional_state = organism.emotions.describe_emotional_state()
    
    # Step 5: Create the coordinated prompt with clear section labels
    final_query = ""
    
    # Add episodic memory section if available
    if reflections:
        reflection = reflections[0]
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
        
        # Add episodic memory section
        final_query += """
=== EPISODIC MEMORY ===
You recall similar conversations with the user, here are the details:

Current Conversation Match: {current_conversation}
Previous Conversations: {previous_convos}
What has worked well: {what_worked}
What to avoid: {what_to_avoid}
""".format(
            current_conversation=current_conversation,
            previous_convos=' | '.join(previous_convos),
            what_worked=' '.join(what_worked),
            what_to_avoid=' '.join(what_to_avoid)
        )
    
    # Add procedural guidelines section
    final_query += """
=== PROCEDURAL GUIDELINES ===
{procedural_memory}
""".format(procedural_memory=procedural_memory)
    
    # Add knowledge context section
    final_query += """
=== KNOWLEDGE CONTEXT ===
{knowledge_context}
""".format(knowledge_context=knowledge_context)
    
    # Add emotional state section if available
    if emotional_state:
        final_query += """
=== EMOTIONAL STATE ===
The following is your emotional state. Use it to influence your tone and response.
{emotional_state}
""".format(emotional_state=emotional_state)
    
    # Add user message section
    final_query += """
=== USER MESSAGE ===
{query}
""".format(query=query)
    
    # Add instructions for using the context
    final_query += """
=== INSTRUCTIONS ===
Use all the above information, as well as previous messages in this conversation, as context for your response to the user.
"""
    
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
    organism=None,  # Add organism parameter
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500
) -> str:
    """Process a user query using RAG and return AI response."""
    client = OpenAI()
    working_memory = memory.get_working_memory()
    
    # Initialize with system message if first interaction
    if not working_memory.get_messages():
        working_memory.initialize_with_system_message(
            "You converse with the human/user."
        )
    
    # Store original query in working memory
    working_memory.add_user_message(query)
    
    # Use prompt coordinator to create the comprehensive prompt
    messages = prompt_coordinator(
        query,
        df,
        memory,
        organism,  # Pass the organism
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
            
            analyze_user_message(user_query, organism.emotions)

            answer = ask(user_query, knowledge_df, memory, organism)  # Pass the organism
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
        
        # Save emotional state if the organism has emotions
        if hasattr(organism, 'emotions') and organism.emotions:
            print("Saving emotional state...")
            organism.emotions.save_to_file()

