## Introduction
Sentiment analysis is a natural language processing (NLP) technique used to determine whether a piece of text expresses a positive, negative, or neutral sentiment.  
In this project, we perform sentiment analysis on **IMDB movie reviews** to classify them as either **positive** or **negative**.

---

## Objective
- To predict the sentiment of movie reviews based on text data.
- To compare the performance of multiple machine learning models.
- To gain insights into the most effective preprocessing and feature extraction techniques for text classification.

---

## Dataset
- Source: [IMDB Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Size: 50,000 reviews (25,000 positive, 25,000 negative)
- Format: CSV file with two columns:
  - `review`: Text of the movie review
  - `sentiment`: Label (`positive` or `negative`)

---

## Preprocessing
Text preprocessing is an essential step to clean the dataset and prepare it for feature extraction:
1. Convert text to lowercase.
2. Remove HTML tags and special characters.
3. Remove stopwords (common words like "the", "is", etc.).
4. Perform lemmatization to reduce words to their base form.
5. Tokenize the text into words.
6.

 ---
## Machine Learning Models

Several machine learning algorithms can be used for sentiment analysis:
Logistic Regression (LR)
Naive Bayes (MultinomialNB)
Random Forest Classifier (RFC)
Support Vector Machine (SVM)
Gradient Boosting Classifier (GBC)
Evaluation Metrics
Accuracy: Percentage of correct predictions.
Precision: Correct positive predictions out of all predicted positives.
Recall: Correct positive predictions out of all actual positives.
F1-Score: Harmonic mean of precision and recall.
Confusion Matrix: Table showing True Positive, True Negative, False Positive, and False Negative.


 ---

 
## Results
| Model                    | Accuracy | Precision | Recall | F1-Score |
| ------------------------ | -------- | --------- | ------ | -------- |
| Logistic Regression      | 0.88     | 0.87      | 0.89   | 0.88     |
| Multinomial Naive Bayes  | 0.86     | 0.85      | 0.87   | 0.86     |
| Random Forest Classifier | 0.87     | 0.86      | 0.87   | 0.86     |
| SVM                      | 0.88     | 0.88      | 0.88   | 0.88     |
| Gradient Boosting        | 0.87     | 0.86      | 0.87   | 0.87     |




Note: Results may vary depending on preprocessing and vectorization.

----

## Conclusion

Sentiment analysis using machine learning models can effectively classify movie reviews.
Logistic Regression and SVM performed best for this dataset.
Proper text preprocessing and feature extraction are key for improving model performance.
Future Work
Implement deep learning models like LSTM, GRU, or BERT for better accuracy.
Use hyperparameter tuning to optimize model performance.
Apply ensemble techniques combining multiple models for improved results.
Build a real-time web application to predict sentiment of new reviews.


---




## References

IMDB Movie Reviews Dataset on Kaggle

Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O’Reilly Media.

Scikit-learn Documentation: https://scikit-learn.org

NLTK Documentation: https://www.nltk.org


-----
Example:

```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
corpus = []

for review in reviews['review']:
    text = re.sub('[^a-zA-Z]', ' ', review)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(text))


##Feature Extraction

To convert text into numerical features, we use Bag of Words (BoW) or TF-IDF Vectorization:

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Example with CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()


CountVectorizer creates a vocabulary of most frequent words and represents each review as a vector.

TfidfVectorizer gives higher weight to important words and reduces the impact of common words.
