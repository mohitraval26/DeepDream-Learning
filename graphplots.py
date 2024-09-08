import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('punkt', download_dir='C:/Users/Dhruv/nltk_data')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('dreamdata.csv')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
sid = SentimentIntensityAnalyzer()

# Text Preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Removing stop words and non-alphabetic words
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

df['text_cleaned'] = df['content'].apply(preprocess_text)

# Topic Modeling using LDA
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['text_cleaned'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda, vectorizer.get_feature_names_out(), 10)

# Sentiment Analysis
def analyze_sentiment(text):
    return sid.polarity_scores(text)

df['sentiment'] = df['text_cleaned'].apply(analyze_sentiment)
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df['positive'] = df['sentiment'].apply(lambda x: x['pos'])
df['negative'] = df['sentiment'].apply(lambda x: x['neg'])
df['neutral'] = df['sentiment'].apply(lambda x: x['neu'])

# Plotting Emotional Trends
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='date', y='compound', label='Compound Sentiment')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Trends Over Time')
plt.legend()
plt.show()

# Word Cloud for Dream Content
text = ' '.join(df['text_cleaned'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Dream Content')
plt.show()

# Emotional Distribution
emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
plt.figure(figsize=(12, 8))
df[emotions].mean().plot(kind='bar')
plt.xlabel('Emotion')
plt.ylabel('Average Score')
plt.title('Average Emotion Scores in Dreams')
plt.show()

# Example of classification (if needed)
# Use emotion scores to predict a categorical variable
X_emotion = df[emotions]
y = df['description']  # Using 'description' as the target variable

# Scaling
scaler = StandardScaler()
X_emotion_scaled = scaler.fit_transform(X_emotion)

# Classification
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_emotion_scaled, y)

# Predictions (for example purposes)
predictions = clf.predict(X_emotion_scaled)
print(predictions)
