import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize NLP tools
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

# Load dataset
df = pd.read_csv('dreamdata.csv')

# Text Preprocessing Function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

df['text_cleaned'] = df['content'].apply(preprocess_text)

# Topic Modeling
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['text_cleaned'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_features = lda.fit_transform(X)

# Add topic probabilities to the dataframe
df = pd.concat([df, pd.DataFrame(lda_features, columns=[f'Topic_{i}' for i in range(lda_features.shape[1])])], axis=1)

# Sentiment Analysis
def analyze_sentiment(text):
    return sid.polarity_scores(text)

df['sentiment'] = df['text_cleaned'].apply(analyze_sentiment)
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])

# Prepare data for classification
X_features = df[[f'Topic_{i}' for i in range(lda_features.shape[1])] + ['compound']]
y = df[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'negative', 'positive']]  # Using emotion columns for prediction

# Scaling
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_features_scaled, y)

# Emotion labels
emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'negative', 'positive']

# Function to classify new dream input
def classify_dream(user_input):
    user_input_cleaned = preprocess_text(user_input)
    user_input_vector = vectorizer.transform([user_input_cleaned])
    user_input_topics = lda.transform(user_input_vector)
    user_input_sentiment = analyze_sentiment(user_input)
    
    # Ensure user_input_sentiment['compound'] is a 2D array
    sentiment_feature = np.array([[user_input_sentiment['compound']]])
    
    # Concatenate topic features and sentiment feature
    features = np.hstack([user_input_topics, sentiment_feature])
    
    # Ensure feature dimensions match
    if features.shape[1] != X_features.shape[1]:
        raise ValueError(f"Feature dimension mismatch: Expected {X_features.shape[1]}, but got {features.shape[1]}")
    
    features_scaled = scaler.transform([features.flatten()])  # Flatten to 1D for scaling
    
    prediction = clf.predict(features_scaled)
    
    # Convert prediction to a dictionary
    prediction_dict = dict(zip(emotion_labels, prediction[0]))
    
    return prediction_dict

# Take user input
user_input = input("Enter your dream description: ")

# Classify and print results
dream_features = classify_dream(user_input)

print("Dream features:")
for emotion, value in dream_features.items():
    print(f"{emotion}: {value}")
