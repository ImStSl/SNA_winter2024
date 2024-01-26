import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

df = pd.read_csv('../dataset/dataset.csv', low_memory=False)

# Sample data
tags_list = df['tags'].dropna().tolist()

tags_content = ' '.join(tags_list)

# Tokenize the content
tags_tokens = word_tokenize(tags_content)

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
filtered_tags_tokens = [word for word in tags_tokens if word.isalnum() and word.lower() not in stop_words]

# Count the frequency of each token
tags_word_freq = Counter(filtered_tags_tokens)

# Select the top 25 keywords
top_keywords = [word for word, freq in tags_word_freq.most_common(25)]

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()

positive_keywords = 0
negative_keywords = 0
neutral_keywords = 0
for keyword in top_keywords:
    sentiment_score = sia.polarity_scores(keyword)['compound']
    if sentiment_score >= 0.05:
        positive_keywords += 1
    elif sentiment_score <= -0.05:
        negative_keywords += 1
    else:
        neutral_keywords += 1


print(f"Top 25 Keywords (excluding stop words): {top_keywords}")
print(f"Positive Keywords: {positive_keywords}")
print(f"Negative Keywords: {negative_keywords}")
print(f"Neutral Keywords: {neutral_keywords}")