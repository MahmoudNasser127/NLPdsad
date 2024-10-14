#Import Required Libraries
import pandas as pd
import nltk
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv('C:\\Users\\Mahmo\\Coding\\NLP Project\\IMDB Dataset.csv')

df.describe()

# %%
df = df.drop_duplicates()


# %%
#Converting to lowercase
df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
#Removing remove non-word and non-whitespace characters
df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)
# Convert 'text' column to string data type
df['review'] = df['review'].astype(str)  
  # Tokenization
df['tokens'] = df['review'].apply(nltk.word_tokenize)

# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])


# Apply stemming and lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

df['tokens_stemmed'] = df['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])
df['tokens_lemmatized'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])


df['processed_review'] = df['tokens_lemmatized'].apply(lambda x: ' '.join(x))


# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_review'])

# Convert 'sentiment' column to binary (assuming sentiment is 'positive' or 'negative')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment'], test_size=0.2, random_state=42)

# Model Training
model = SVC(kernel='linear', C=1)  # You can adjust parameters based on the performance
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

import pickle

# Saving the model as a pickle file
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Saving the TfidfVectorizer as a pickle file
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved as pickle files.")





