#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

# Load the dataset (update 'path/to/your/file.csv' with actual file location)
df = pd.read_csv('/Users/samithlakka/Desktop/Capstone 3/WELFake_Dataset.csv')

# Display basic info and check for missing values
print(df.info())
print(df.head())
print(df.isnull().sum())
print(f"Duplicates: {df.duplicated().sum()}")


# In[6]:


# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Drop rows where title or text is missing
df.dropna(subset=['title', 'text'], inplace=True)

# Convert text columns to lowercase
df['title'] = df['title'].astype(str).str.lower()
df['text'] = df['text'].astype(str).str.lower()

print("Data cleaned. Missing values removed.")


# In[7]:


import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Text cleaning: removes punctuation, stopwords, and applies lemmatization."""
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization
    return ' '.join(words)

# Apply text cleaning to 'text' column
df['cleaned_text'] = df['text'].apply(preprocess_text)

print("Text preprocessing complete.")


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Convert cleaned text into numerical representation
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 words
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Convert to DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate with original dataset
df_final = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

print("TF-IDF feature engineering complete.")


# In[9]:


from sklearn.model_selection import train_test_split

# Define X (features) and y (target label)
X = df_final.drop(columns=['title', 'text', 'cleaned_text', 'label'])  # Drop original text columns
y = df_final['label']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split complete. Ready for modeling.")


# In[ ]:




