from textblob import TextBlob
import datetime

def analyze_user_message(user_message, emotions_obj):
    """
    Analyze a user message and update emotions based on sentiment.
    
    Args:
        user_message (str): The user's message
        emotions_obj: An instance of the Emotions class
        
    Returns:
        dict: The updated emotional state
    """
    # Analyze sentiment with TextBlob
    blob = TextBlob(user_message)
    
    # polarity: -1 is negative, 0 is neutral, 1 is positive
    # subjectivity: 0 is objective (factual), 1 is subjective (opinion)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Simple emotion adjustments based on polarity and subjectivity
    if abs(polarity) < 0.2:
        # Message is neutral, no emotion change
        return emotions_obj.current
    
    # Get current emotions as a working copy
    emotions = emotions_obj.current.copy()
    
    # Positive sentiment increases joy, decreases sadness and anger
    if polarity > 0:
        emotions["joy"] = min(1.0, emotions["joy"] + 0.1 * polarity)
        emotions["sadness"] = max(0.0, emotions["sadness"] - 0.05 * polarity)
        emotions["anger"] = max(0.0, emotions["anger"] - 0.05 * polarity)
        
        # High subjectivity with positive polarity can increase surprise
        if subjectivity > 0.5:
            emotions["surprise"] = min(1.0, emotions["surprise"] + 0.05 * subjectivity)
    
    # Negative sentiment increases sadness, anger, and possibly fear
    else:
        emotions["sadness"] = min(1.0, emotions["sadness"] + 0.1 * abs(polarity))
        
        # Higher subjectivity with negative polarity increases anger more than sadness
        if subjectivity > 0.5:
            emotions["anger"] = min(1.0, emotions["anger"] + 0.1 * abs(polarity) * subjectivity)
        
        # Lower subjectivity with negative polarity may indicate fear rather than anger
        else:
            emotions["fear"] = min(1.0, emotions["fear"] + 0.05 * abs(polarity))
            
        emotions["joy"] = max(0.0, emotions["joy"] - 0.05 * abs(polarity))
    
    # Round values for readability
    emotions = {k: round(v, 2) for k, v in emotions.items()}
    
    # Update the emotions object
    emotions_obj.current = emotions
    emotions_obj.last_updated = datetime.datetime.now()
    
    # Save if we have a file path
    if emotions_obj.file_path:
        emotions_obj.save_to_file()
    
    return emotions