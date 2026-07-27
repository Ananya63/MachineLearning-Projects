import numpy as np
import pandas as pd
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
# Load and preprocess the diabetes dataset
data = pd.read_csv("/home/dst-fist/Desktop/Ananya_ML_LAB/Naive/Hotel_Reviews.csv")
data.loc[:, 'Positive_Review'] = data.Positive_Review.apply(lambda x: x.replace('No Positive', ''))
data.loc[:, 'Negative_Review'] = data.Negative_Review.apply(lambda x: x.replace('No Negative', ''))
new_data = []
for index, row in data.iterrows():
    # Combine positive and negative reviews into the new column
    # If the review is from Positive_Review, label it with 1, else label with 0 for Negative_Review
    if pd.notnull(row['Positive_Review']):
        new_data.append([row['Positive_Review'], 1])
    if pd.notnull(row['Negative_Review']):
        new_data.append([row['Negative_Review'], 0])

new_df = pd.DataFrame(new_data, columns=['Review', 'Label'])
shuffled_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split
X_train, X_test = train_test_split(shuffled_df, test_size=0.2, random_state=42)

import nltk
nltk.download('stopwords')

def clean_text(text):
    # lower text
    text = text.lower()
    
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]

    #stemming
    ps=PorterStemmer()
    text = [ps.stem(x) for x in text]
    
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    return(text)

X_train["review_clean"] = X_train["Review"].apply(lambda x: clean_text(x))

# Step 1: Create Vocabulary and Word Counts for each Class (Positive and Negative)
def build_class_word_counts(df):
    # Initialize counters for positive and negative reviews
    positive_reviews = df[df['Label'] == 1]['review_clean']
    negative_reviews = df[df['Label'] == 0]['review_clean']
    positive_reviews = [word for review in positive_reviews for word in review]
    negative_reviews = [word for review in negative_reviews for word in review]
    #count word frequencies
    positive_word_counts = Counter(positive_reviews)
    negative_word_counts = Counter(negative_reviews)
    #print("pro word count=", positive_reviews)
    # Total words in each class
    total_positive_words = sum(positive_word_counts.values())
    total_negative_words = sum(negative_word_counts.values())
    
    return positive_word_counts, negative_word_counts, total_positive_words, total_negative_words

# Step 2: Apply Laplace Smoothing
def laplace_smoothing(positive_word_counts, negative_word_counts, total_positive_words, total_negative_words, vocab_size):
    wordsp = list(positive_word_counts.keys())
    wordsn = list(negative_word_counts.keys())
    # For each word in the text, calculate the Laplace-smoothed probability for the positive class
    for word in wordsp:
        word_count = positive_word_counts[word]
        # Apply Laplace smoothing formula:
        smoothed_probabilities_p[word] = (word_count + 1) / (total_positive_words + vocab_size)
    for word in wordsn:
        word_count = negative_word_counts[word]
        # Apply Laplace smoothing formula:
        smoothed_probabilities_n[word] = (word_count + 1) / (total_negative_words + vocab_size)

# Initialize dictionary to store the smoothed probabilities for each word
smoothed_probabilities_p = {}
smoothed_probabilities_n = {}
# Build word counts for positive and negative classes
positive_word_counts, negative_word_counts, total_positive_words, total_negative_words = build_class_word_counts(X_train)

# Vocabulary size (unique words across both classes)
vocab = set(" ".join(review) for review in X_train['review_clean'])
vocab_size = len(vocab)

# Calculate smoothed probabilities for each review in shuffled_df
laplace_smoothing(positive_word_counts, negative_word_counts, total_positive_words, total_negative_words, vocab_size)

X_test["review_clean"] = X_test["Review"].apply(lambda x: clean_text(x))
reviews = X_test['review_clean']
cl=[]
p_norm=[]
n_norm=[]
label_predict=[]
for i in reviews:
    p = 0.5
    n = 0.5
    word_found = False
    for word in i:
        if word in smoothed_probabilities_n and word in smoothed_probabilities_p:
            p *= smoothed_probabilities_p[word]
            n *= smoothed_probabilities_n[word]
            word_found = True  # Mark that we processed at least one valid word
    
    if (p+n)==0:
        # Handle case where no words were found in the dictionaries
        p_normalized = 0.5
        n_normalized = 0.5
    else:
        # Normalize probabilities
        p_normalized = p / (p + n)
        n_normalized = n / (p + n)
    p_norm.append(p_normalized)
    n_norm.append(n_normalized)
    if p_norm>=n_norm:
        cl.append("positive")
        label_predict.append(1)
    elif p_norm<n_norm:
        cl.append("negative")
        label_predict.append(0)

# Create a DataFrame
df_results = pd.DataFrame({
    "Review": X_test["Review"],
    "P_Norm": p_norm,
    "N_Norm": n_norm,
    "Class": cl,
    "Label_Predict": label_predict
})
df_results['Class'] = df_results.apply(lambda row: 'positive' if row['P_Norm'] > row['N_Norm'] else 'negative', axis=1)
df_results['Label_Predict'] = df_results.apply(lambda row: 1 if row['P_Norm'] > row['N_Norm'] else 0, axis=1)

#df_results.head(20)
df_results.to_csv('df_results.csv', index=False)

true_labels = X_test["Label"].values  # True labels from X_test
predicted_labels = df_results["Label_Predict"].values  # Predicted labels from df_results

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.2f}")
