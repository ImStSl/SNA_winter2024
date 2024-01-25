import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


df = pd.read_csv('../dataset/usdataset.csv', low_memory=False)

# Sample data
titles_list = df['title'].dropna().tolist()
descriptions_list = df['description'].dropna().tolist()
tags_list = df['tags'].dropna().tolist()

# Combine titles, descriptions, and tags into a single string
combined_content = ' '.join(titles_list + descriptions_list + tags_list)

# Tokenize the content
content_tokens = word_tokenize(combined_content)

additional_stop_words = ['https', 'http', 'YouTube', 'video', 'get', 'channel', 'na', 'new', 'Instagram',
                         'us', 'like', 'Official', '2024', 'Twitter', 'Facebook', 'de', 'vs', 'Video', 'None',
                         'first', 'one', 'Watch','youtube', 'official', 'instagram', 'subscribe', 'watch',
                         'twitter', 'world', 'facebook', 'news', 'best', 'follow', 'videos', 'none', 'show',
                         'live', 'free', 'love', 'time','check', 'content', 'merch', 'tiktok', 'make', 'full',
                         'use', 'go', 'things', 'take', 'every', 'got', 'find', 'know', 'highlights', 'day',
                         '2','latest', 'today', 'company', 'last', 'back', 'want', '10', 'visit', 'code', '5',
                         'yes', 'shop', 'social', 'life', 'night', 'see', 'people', 'website', 'favorita',
                         'play',  'en', 'link', 'way', 'thanks', 'stream', 'week', 'podcast', 'much', 'el',
                         'good', 'drag', 'also','could', 'app', 'la', 'try', 'enough', 'home', 'never', 'still',
                         'director', '3', 'episode', 'let', 'sent',  'impact', 'tumbler', 'white', 'green', 'game'
                         , '21', 'du', 'football', 'generals', 'press','grande', 'fans']

# Remove stop words
stop_words = set(stopwords.words('english') + additional_stop_words)
filtered_content_tokens = [word for word in content_tokens if word.isalnum() and word.lower() not in stop_words]

# Join the remaining tokens into phrases for sentiment analysis
phrases = ' '.join(filtered_content_tokens)

# Sentiment Analysis with VADER
sia = SentimentIntensityAnalyzer()
sentiment_score = sia.polarity_scores(phrases)

# Print results
print("Sentiment Analysis Results:")
print(sentiment_score)
