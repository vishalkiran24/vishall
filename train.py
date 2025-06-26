import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_text, tfidf_features

# Load data
df = pd.read_csv('resume_data.csv')
df['clean_text'] = df['resume_text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['category'], test_size=0.2, random_state=42)

# TF-IDF features
X_train_tfidf, vectorizer = tfidf_features(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Choose classifier
clf = MultinomialNB()  # Or LinearSVC(), LogisticRegression()
clf.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))   
print(confusion_matrix(y_test, y_pred))

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)