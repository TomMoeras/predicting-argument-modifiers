import os  # nopep8
import sys  # nopep8
from pathlib import Path  # nopep8

# Add the 'src' directory to sys.path
current_working_directory = os.getcwd()  # Get the current working directory
src_path = (
    Path(current_working_directory).resolve() / "src"
)  # Construct the path to the 'src' directory
sys.path.append(str(src_path))  # Add 'src' directory to sys.path
import spacy_sentence_bert

from corpus import *
from models.framework import *

# define the corpus path
corpus_path = Path(src_path).resolve() / "corpus" / "md_corpus_ontonotes.pkl"

md_corpus_onto = Corpus.load_corpus(corpus_path)

from sklearn.linear_model import LogisticRegression

# Instantiate the FeatureExtractor object. This object is responsible for transforming raw text data into feature vectors
# that can be used by machine learning models. It does this by applying various feature extraction methods to the text.

feature_extractor = FeatureExtractor(
    feature_methods=[
        # Bag of Words (BOW): This method creates a feature vector that represents the frequency of each word in the text.
        # The dimensionality of the feature vector is equal to the size of the vocabulary. This is a basic "dummy" feature
        # extraction method for text data.
        FeatureExtractor.bow,
        # TF-IDF (Term Frequency-Inverse Document Frequency): This method is similar to BOW, but it weights each word's frequency
        # by its inverse document frequency (i.e., log(total number of documents / number of documents containing the word)).
        # This gives more weight to rare words, which are often more informative than common words.
        FeatureExtractor.tfidf,
        # N-gram: This method creates feature vectors that represent the frequency of each n-gram in the text. An n-gram is a
        # contiguous sequence of n words. For example, in the sentence "I love dogs", the bi-grams are
        # "I love" and "love dogs". N-grams capture some of the contextual and sequential information in the text.
        FeatureExtractor.ngram,
        # POS (Part of Speech): This method creates feature vectors that represent each part of speech in the
        # text. For example, it can provide information on how many nouns, verbs, adjectives, etc. are in the text and how they are
        # related to modifier roles.
        FeatureExtractor.pos,
        # NER (Named Entity Recognition): This method creates feature vectors that represent each type of named
        # entity in the text. Named entities are things like people's names, company names, locations, product names, etc.
        FeatureExtractor.ner,
        # Dependency Parsing: This method extracts the dependency structure of the sentence and represents it as a feature vector.
        FeatureExtractor.get_dependency_features,
        # Constituency Parsing: This method extracts the constituency structure of the sentence (i.e., the parse tree) and
        # represents it as a feature vector.
        FeatureExtractor.get_constituency_features,
    ],
    corpus=md_corpus_onto,  # The corpus of text data on which the feature extraction methods will be applied
)

# Instantiate the PamModel object. This object is a model that predicts the modifier role of a phrase (in a
# sentence), based on the feature vectors created by the FeatureExtractor. The machine learning algorithm used is
# logistic regression, which is a suitable choice for multiclass classification tasks.

model = PamModel(
    model=LogisticRegression(
        max_iter=300
    ),  # Using Logistic Regression for fitting the features.
    feature_extractor=feature_extractor,  # The feature extractor that will be used to transform the raw text data into feature vectors.
    corpus=md_corpus_onto,  # The corpus of text data on which the model will be trained and tested.
    train_data="train",  # The part of the corpus to use for training. This is a set of labeled data.
    test_data="dev",  # The part of the corpus to use for testing. This is a set of labeled data.
)

# Train the model. This involves applying the feature extractor to the training data, and then training the logistic regression
# model on the resulting feature vectors and labels. The trained model can then be used to make predictions on new, unlabeled data.
model.train()

# save model
model.save_model("md_pam_model.pkl")
