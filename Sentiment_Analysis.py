#############################
# Business Problem
#############################
# Kozmos, which focuses on home textiles and casual wear production and sells its products through Amazon, aims to increase its sales
# by analyzing the comments on its products and improving its features based on the received complaints. In line with this goal,
# sentiment analysis will be performed on the comments, they will be labeled, and a classification model will be created with the labeled data

#############################
# Dataset Story
#############################
# The dataset consists of variables indicating the comments made on a specific product group, the comment title, the number of stars,
# and the number of people who found the comment helpful.

# Star      : Number of stars given to the product
# HelpFul   : Number of people who found the comment helpful
# Title     : Title given to the comment content, short comment
# Review    : Comment made on the product



#############################
# 1. Text Pre-Processing
#############################

# !pip install nltk
# !pip install textblob
# !pip install wordcloud
# !pip install openpyxl

from warnings import filterwarnings
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from textblob import Word
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

df = pd.read_excel("NLP/Datasets/amazon.xlsx")
df.head()

###############################
# Normalizing Case Folding
###############################
df['Review'] = df['Review'].str.lower()

###############################
# Punctuations
###############################
df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# regular expression

###############################
# Numbers
###############################
df['Review'] = df['Review'].str.replace('\d', '')

###############################
# Stopwords
###############################
import nltk
# nltk.download('stopwords')

sw = stopwords.words('english')

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords
###############################

temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()
temp_df

drops = temp_df[-1000:]

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

###############################
# Lemmatization
###############################

# nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head(20)

#############################
# 2. Text Visualization
#############################

###############################
# Calculation of Term Frequencies
###############################

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

###############################
# Barplot
###############################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

###############################
# Wordcloud
###############################

text = " ".join(i for i in df["Review"])


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")


##################################################
# 3. Sentiment Analysis
##################################################

df["Review"].head()

# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")


df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()


# NOTE: By labeling the comments with SentimentIntensityAnalyzer, the dependent variable for the comment classification machine learning model has been created.

df.groupby("sentiment_label")["Star"].mean()

df.head()


###############################
# 4. Feature Engineering
###############################


df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["Review"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###############################
# TF-IDF
###############################
tf_idf_word_vectorizer = TfidfVectorizer().fit(X_train)
X_train_tf_idf_word = tf_idf_word_vectorizer.transform(X_train)
X_test_tf_idf_word = tf_idf_word_vectorizer.transform(X_test)



###############################
# 5. Logistic Regression
###############################

log_model = LogisticRegression().fit(X_train_tf_idf_word, y_train)


y_pred = log_model.predict(X_test_tf_idf_word)

print(classification_report(y_pred, y_test))

cross_val_score(log_model, X_test_tf_idf_word, y_test, cv=5).mean()

random_review = pd.Series(df["Review"].sample(1).values)
New_review = TfidfVectorizer().fit(X_train).transform(random_review)
pred = log_model.predict(New_review)
print(f'Review:  {random_review[0]} \n Prediction: {pred}')

###############################
# 6. Random Forest
###############################

rf_model = RandomForestClassifier().fit(X_train_tf_idf_word, y_train)
cross_val_score(rf_model, X_test_tf_idf_word, y_test, cv=5, n_jobs=-1).mean()

