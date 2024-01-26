import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


df = pd.read_csv('../dataset/dataset.csv', low_memory=False)

# Sample data
titles_list = df['title'].dropna().tolist()
descriptions_list = df['description'].dropna().tolist()
tags_list = df['tags'].dropna().tolist()

# Combine titles, descriptions, and tags into a single string
combined_content = ' '.join(titles_list + descriptions_list + tags_list)

# Tokenize the content
content_tokens = word_tokenize(combined_content)


# Remove stop words
stop_words = set(stopwords.words('english') )
filtered_content_tokens = [word for word in content_tokens if word.isalnum() and word.lower() not in stop_words]

# Join the remaining tokens into phrases for sentiment analysis
phrases = ' '.join(filtered_content_tokens)

# Sentiment Analysis with VADER
sia = SentimentIntensityAnalyzer()
sentiment_score = sia.polarity_scores(phrases)

# Print results
print("Sentiment Analysis Results:")
print(sentiment_score)
