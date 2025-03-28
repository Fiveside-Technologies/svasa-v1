from textblob import TextBlob

text = "I hate this so much!"
blob = TextBlob(text)

# polarity: -1 is negative, 0 is neutral, 1 is positive
# subjectivity: 0 is objective (factual), 1 is subjective (opinion)
sentiment = blob.sentiment
print(sentiment)