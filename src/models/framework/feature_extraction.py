import spacy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import benepar  # this needs to be imported for the constituency parsing to work
from nltk import Tree

import logging

logging.getLogger().setLevel(logging.WARNING)
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

from models.framework.feature_extraction_helpers import *


class FeatureExtractor:
    """
    This class is responsible for extracting a variety of features from the input data.

    It includes various methods to transform the data into features using techniques such as bag-of-words,
    TF-IDF, n-grams, part-of-speech (POS) tagging, named entity recognition (NER), dependency parsing, constituency parsing,
    and the use of sentence-BERT embeddings.

    Attributes:
        feature_methods (list): A list of methods that define how to extract features.
        corpus (Corpus): The corpus of data to work with.
        nlp (Spacy.Language): A SpaCy language model, loaded with the specified name.
        nlp_benepar (Spacy.Language): A SpaCy language model, loaded with the Benepar constituency parsing model.
        sbert_nlp (Spacy.Language): A SpaCy language model, loaded with the sentence-BERT model.
        vectorizer (CountVectorizer or TfidfVectorizer): Vectorizer for extracting text features.
        pos_encoder (OneHotEncoder): Encoder for one-hot encoding of POS tags.
        dep_encoder (OneHotEncoder): Encoder for one-hot encoding of dependency parse tree features.
        ner_encoder (OneHotEncoder): Encoder for one-hot encoding of NER tags.
        constituency_encoder (OneHotEncoder): Encoder for one-hot encoding of constituency parse tree features.

    Methods:
        transform(self, data, fit=False): Apply the specified feature extraction methods to the input data.
        bow(self, data, fit=False): Apply a bag-of-words transformation to the input data.
        tfidf(self, data, fit=False, max_features=5000): Apply a TF-IDF transformation to the input data.
        ngram(self, data, fit=False, ngram_range=(1, 2), max_features=10000): Apply an n-gram transformation to the input data.
        ngram_lemma(self, data, fit=False, ngram_range=(1, 2), max_features=10000): Apply an n-gram transformation to the lemmatized input data.
        pos(self, data, fit=False): Encode the POS tags of the input data.
        get_dependency_features(self, data, fit=False): Extract and encode the features from the dependency parse trees of the input data.
        ner(self, data, fit=False): Extract and encode the NER tags of the input data.
        get_constituency_features(self, data, fit=False): Extract and encode the features from the constituency parse trees of the input data.
        get_sentence_bert_features(self, data, fit=False): Apply the sentence-BERT model to the input data to obtain sentence embeddings.
    """

    UNPROCESSED = "unprocessed"
    MEMORY_ERROR_MSG = "MemoryError encountered in {}. Retrying with less data..."
    NOT_FITTED_ERROR_MSG = (
        "Encoder/Vectorizer not fitted yet. Please fit with training data first."
    )

    def __init__(self, feature_methods, corpus, spacy_model_name="en_core_web_sm"):
        self.feature_methods = feature_methods
        self.vectorizer = None
        self.pos_encoder = None
        self.dep_encoder = None
        self.ner_encoder = None
        self.constituency_encoder = None
        self.corpus = corpus
        self.nlp = spacy.load(spacy_model_name)
        self.nlp_benepar = spacy.load(spacy_model_name)
        # Add the constituency parsing model to the pipe
        self.nlp_benepar.add_pipe("benepar", config={"model": "benepar_en3"})
        # Load sentence-BERT model
        self.sbert_nlp = spacy_sentence_bert.load_model("en_stsb_distilbert_base")

    def transform(self, data, fit=False):
        """
        Apply the specified feature extraction methods to the input data.

        Parameters:
        data (pandas.DataFrame): Input data to extract features from.
        fit (bool, optional): If True, the feature extraction methods will be fitted to the data.
                            This should be True for the initial training data, and False for any subsequent data.
                            Defaults to False.

        Returns:
        pandas.DataFrame: The input data, transformed into features.
        """

        # Initialize an empty list to hold the feature matrices produced by each method
        features = []

        # Loop over each feature extraction method specified in self.feature_methods
        for method in self.feature_methods:
            # Try to apply the method to the data
            try:
                # If fit=True, the method will be fitted to the data before transforming it
                # Otherwise, the method will only transform the data
                # The resulting feature matrix is appended to the list of features
                features.append(method(self, data, fit))

            # If applying the method to the data causes a MemoryError
            except MemoryError:
                # Print a message indicating that a MemoryError was encountered
                print(self.MEMORY_ERROR_MSG.format(method.__name__))

                # Sample a random subset of the data, with size equal to 50% of the original data size
                # The random_state parameter ensures reproducibility of the sampling
                data_sampled = data.sample(frac=0.5, random_state=1)

                # Try to apply the method to the sampled subset of the data
                # The resulting feature matrix is appended to the list of features
                features.append(method(self, data_sampled, fit))

        # Concatenate the list of feature matrices along the column axis (i.e., side by side)
        # This results in a single DataFrame where each column represents a different feature
        # and each row corresponds to the same row in the input data
        return pd.concat(features, axis=1)

    def bow(self, data, fit=False):
        """
        Generate Bag of Words (BoW) features for the input data.

        Parameters:
        data (pandas.DataFrame): Input data to extract features from.
                                It is expected to contain a column named 'string' with text data.
        fit (bool, optional): If True, the CountVectorizer will be fitted to the data.
                            This should be True for the initial training data, and False for any subsequent data.
                            Defaults to False.

        Returns:
        pandas.DataFrame: The BoW features for the input data.
        """

        # If fit is True, the vectorizer needs to be initialized and fitted
        if fit:
            # Initialize a CountVectorizer with a limit on the number of features
            # Here, the vectorizer is limited to the 2000 most frequent words for memory efficiency
            self.vectorizer = CountVectorizer(max_features=2000)

            # Fit the vectorizer to the data and transform the data to BoW features
            bow_features = self.vectorizer.fit_transform(data["string"])
        else:
            # If fit is False, the vectorizer should already be initialized and fitted
            # If it's not, raise an exception
            if self.vectorizer is None:
                raise Exception(
                    "Vectorizer not fitted yet. Please fit vectorizer with training data first."
                )

            # Transform the data to BoW features using the already fitted vectorizer
            bow_features = self.vectorizer.transform(data["string"])

        # Get the feature names (words) from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Convert the sparse feature matrix to a DataFrame, with words as columns
        # The indices of the DataFrame match the indices of the input data
        return pd.DataFrame.sparse.from_spmatrix(
            bow_features, columns=feature_names, index=data.index
        )

    def tfidf(self, data, fit=False, max_features=5000):
        """
        Generate TF-IDF features for the input data.

        Parameters:
        data (pandas.DataFrame): Input data to extract features from.
                                It is expected to contain a column named 'string' with text data.
        fit (bool, optional): If True, the TfidfVectorizer will be fitted to the data.
                            This should be True for the initial training data, and False for any subsequent data.
                            Defaults to False.
        max_features (int, optional): Maximum number of features (i.e., words) to consider. Defaults to 5000.

        Returns:
        pandas.DataFrame: The TF-IDF features for the input data.
        """

        # If fit is True, the vectorizer needs to be initialized and fitted
        if fit:
            # Initialize a TfidfVectorizer with a limit on the number of features
            self.vectorizer = TfidfVectorizer(max_features=max_features)

            # Fit the vectorizer to the data and transform the data to TF-IDF features
            tfidf_features = self.vectorizer.fit_transform(data["string"])
        else:
            # If fit is False, the vectorizer should already be initialized and fitted
            # If it's not, raise an exception
            if self.vectorizer is None:
                raise Exception(
                    "Vectorizer not fitted yet. Please fit vectorizer with training data first."
                )

            # Transform the data to TF-IDF features using the already fitted vectorizer
            tfidf_features = self.vectorizer.transform(data["string"])

        # Get the feature names (words) from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Convert the sparse feature matrix to a DataFrame, with words as columns
        # The indices of the DataFrame match the indices of the input data
        return pd.DataFrame.sparse.from_spmatrix(
            tfidf_features, columns=feature_names, index=data.index
        )

    def ngram(self, data, fit=False, ngram_range=(1, 2), max_features=10000):
        """
        Perform a TF-IDF feature extraction with n-gram support on the input data.

        Parameters:
        data (pandas.DataFrame): The input data to extract features from.
        fit (bool, optional): If True, the vectorizer will be fitted to the data.
                            This should be True for the initial training data,
                            and False for any subsequent data. Defaults to False.
        ngram_range (tuple, optional): The range of n-gram values to use. Defaults to (1, 2).
        max_features (int, optional): The maximum number of features to consider.
                                    Defaults to 10000.

        Returns:
        pandas.DataFrame: A DataFrame where each column represents a feature (n-gram)
                        and each row corresponds to the same row in the input data.
        """

        # If fit=True, fit the vectorizer to the data
        if fit:
            # Create a TfidfVectorizer object with specified n-gram range and maximum features
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range, max_features=max_features
            )
            # Fit the vectorizer to the data and transform the data into n-gram features
            ngram_features = self.vectorizer.fit_transform(data["string"])
        else:
            # If fit=False and the vectorizer has not been fitted yet, raise an Exception
            if self.vectorizer is None:
                raise Exception(
                    "Vectorizer not fitted yet. Please fit vectorizer with training data first."
                )
            # If the vectorizer has already been fitted, transform the data into n-gram features
            ngram_features = self.vectorizer.transform(data["string"])

        # Get the names of the features generated by the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Convert the sparse matrix of n-gram features to a DataFrame and return it
        return pd.DataFrame.sparse.from_spmatrix(
            ngram_features, columns=feature_names, index=data.index
        )

    def ngram_lemma(self, data, fit=False, ngram_range=(1, 2), max_features=10000):
        """
        Perform a TF-IDF feature extraction with n-gram support on the lemmatized input data.

        Parameters:
        data (pandas.DataFrame): The input data to extract features from.
        fit (bool, optional): If True, the vectorizer will be fitted to the data.
                            This should be True for the initial training data,
                            and False for any subsequent data. Defaults to False.
        ngram_range (tuple, optional): The range of n-gram values to use. Defaults to (1, 2).
        max_features (int, optional): The maximum number of features to consider.
                                    Defaults to 10000.

        Returns:
        pandas.DataFrame: A DataFrame where each column represents a feature (n-gram)
                        and each row corresponds to the same row in the input data.
        """

        # Initialize a list to store the lemma sequences
        lemma_sequences = []

        # Iterate over each row in the input data
        for _, row in data.iterrows():
            # If the "role_id" field is present and not NaN
            if "role_id" in row and not pd.isna(row["role_id"]):
                # Get the role object associated with the role ID
                role = self.corpus.get_role_by_id(row["role_id"])
                # Create a lemma sequence from the role's document tokens
                lemma_sequence = " ".join([token.lemma_ for token in role.doc])
            # If the "string" field is present and not NaN
            elif "string" in row and not pd.isna(row["string"]):
                # Create a lemma sequence from the row's string
                lemma_sequence = " ".join(
                    [token.lemma_ for token in self.nlp(row["string"])]
                )
            else:
                # If neither "role_id" nor "string" are present, set the lemma sequence to an empty string
                lemma_sequence = ""

            # Append the lemma sequence to the list of lemma sequences
            lemma_sequences.append(lemma_sequence)

        # If fit=True, fit the vectorizer to the lemma sequences
        if fit:
            # Create a TfidfVectorizer object with specified n-gram range and maximum features
            self.ngram_vectorizer = TfidfVectorizer(
                ngram_range=ngram_range, max_features=max_features
            )
            # Fit the vectorizer to the lemma sequences and transform the lemma sequences into n-gram features
            ngram_features = self.ngram_vectorizer.fit_transform(lemma_sequences)
        else:
            # If fit=False and the vectorizer has not been fitted yet, raise an Exception
            if self.ngram_vectorizer is None:
                raise Exception(
                    "Vectorizer not fitted yet. Please fit vectorizer with training data first."
                )
            # If the vectorizer has already been fitted, transform the lemma sequences into n-gram features
            ngram_features = self.ngram_vectorizer.transform(lemma_sequences)

        # Get the names of the features generated by the vectorizer
        feature_names = self.ngram_vectorizer.get_feature_names_out()

        # Convert the sparse matrix of n-gram features to a DataFrame and return it
        return pd.DataFrame.sparse.from_spmatrix(
            ngram_features, columns=feature_names, index=data.index
        )

    def pos(self, data, fit=False):
        """
        Perform a one-hot encoding of the POS (Part-of-Speech) tag sequences in the input data.

        Parameters:
        data (pandas.DataFrame): The input data to extract POS sequences from.
        fit (bool, optional): If True, the one-hot encoder will be fitted to the data.
                            This should be True for the initial training data,
                            and False for any subsequent data. Defaults to False.

        Returns:
        pandas.DataFrame: A DataFrame where each column represents a POS sequence (one-hot encoded)
                        and each row corresponds to the same row in the input data.
        """

        # Initialize a list to store the POS tag sequences
        pos_sequences = []

        # Iterate over each row in the input data
        for _, row in data.iterrows():
            # If the "role_id" field is present and not NaN
            if "role_id" in row and not pd.isna(row["role_id"]):
                # Get the role object associated with the role ID
                role = self.corpus.get_role_by_id(row["role_id"])
                # Create a POS tag sequence from the role's document tokens
                pos_sequence = "_".join([token.pos_ for token in role.doc])
            # If the "string" field is present and not NaN
            elif "string" in row and not pd.isna(row["string"]):
                # Process the string with the language model to get a Doc object
                doc = self.nlp(row["string"])
                # Create a POS tag sequence from the Doc's tokens
                pos_sequence = "_".join([token.pos_ for token in doc])
            else:
                # If neither "role_id" nor "string" are present, set the POS tag sequence to an empty string
                pos_sequence = ""

            # Append the POS tag sequence to the list of POS tag sequences
            pos_sequences.append(pos_sequence)

        # If fit=True, fit the one-hot encoder to the POS tag sequences
        if fit:
            # Create a OneHotEncoder object with handle_unknown="ignore"
            # to ignore any categories not seen during fit when transforming data
            self.pos_encoder = OneHotEncoder(handle_unknown="ignore")
            # Fit the one-hot encoder to the POS tag sequences (reshaped to be 2D)
            self.pos_encoder.fit(np.array(pos_sequences).reshape(-1, 1))

        # Transform the POS tag sequences into one-hot encoded features (reshaped to be 2D)
        pos_features = self.pos_encoder.transform(
            np.array(pos_sequences).reshape(-1, 1)
        )

        # Convert the sparse matrix of one-hot encoded features to a DataFrame and return it
        return pd.DataFrame.sparse.from_spmatrix(
            pos_features,
            columns=self.pos_encoder.get_feature_names_out(),
            index=data.index,
        )

    def ner(self, data, fit=False):
        """
        This method generates Named Entity Recognition (NER) tags for each row in the provided data,
        and then encodes these NER tags into a one-hot encoded format.

        Parameters:
        data (pandas.DataFrame): The input data to extract NER tags from.
        fit (bool, optional): If True, the one-hot encoder will be fitted to the data.
                            This should be True for the initial training data,
                            and False for any subsequent data. Defaults to False.

        Returns:
        pandas.DataFrame: A DataFrame where each column represents a unique NER tag
                        (one-hot encoded) and each row corresponds to the same row in the input data.

        """

        # A list of NER tags is created for each row in the input data
        ner_tags = []
        for _, row in data.iterrows():
            # If the row has a 'role_id' and it is not missing,
            # retrieve the role from the corpus and generate NER tags from it.
            if "role_id" in row and not pd.isna(row["role_id"]):
                role = self.corpus.get_role_by_id(row["role_id"])
                ner_tag = [ent.label_ for ent in role.doc.ents]

            # If the row has a 'string' and it is not missing,
            # generate NER tags from it using the NLP model.
            elif "string" in row and not pd.isna(row["string"]):
                doc = self.nlp(row["string"])
                ner_tag = [ent.label_ for ent in doc.ents]

            # If neither 'role_id' nor 'string' are available or if they are missing,
            # assign an empty list to the NER tags.
            else:
                ner_tag = []

            # Append the list of NER tags for this row to the main list
            ner_tags.append(ner_tag)

        # If fit=True, fit the one-hot encoder to the NER tags
        if fit:
            # Create a OneHotEncoder object with handle_unknown="ignore"
            # to ignore any categories not seen during fit when transforming data.
            self.ner_encoder = OneHotEncoder(handle_unknown="ignore")

            # Transform the list of lists of NER tags into a list of space-separated strings
            ner_tags_transformed = [" ".join(tags) for tags in ner_tags]

            # Fit the one-hot encoder to the transformed NER tags
            self.ner_encoder.fit(
                np.array(ner_tags_transformed).reshape(-1, 1)  # reshape the array
            )

        # Transform the list of lists of NER tags into a list of space-separated strings
        ner_tags_transformed = [" ".join(tags) for tags in ner_tags]

        # Use the one-hot encoder to transform the transformed NER tags into a sparse matrix of one-hot encoded features
        ner_features = self.ner_encoder.transform(
            np.array(ner_tags_transformed).reshape(-1, 1)  # reshape the array
        )

        # Convert the sparse matrix of one-hot encoded features to a DataFrame and return it.
        # The column names are the feature names from the one-hot encoder,
        # and the index is the same as the index of the input data.
        return pd.DataFrame.sparse.from_spmatrix(
            ner_features,
            columns=self.ner_encoder.get_feature_names_out(),
            index=data.index,
        )

    def get_dependency_features(self, data, fit=False):
        """
        This method extracts dependency tree strings from each row in the provided data,
        and then encodes these dependency tree strings into a one-hot encoded format.

        Parameters:
        data (pandas.DataFrame): The input data to extract dependency trees from.
        fit (bool, optional): If True, the one-hot encoder will be fitted to the data.
                            This should be True for the initial training data,
                            and False for any subsequent data. Defaults to False.

        Returns:
        pandas.DataFrame: A DataFrame where each column represents a unique dependency tree string
                        (one-hot encoded) and each row corresponds to the same row in the input data.

        """

        # The 'apply' method is used on the input data to process each row.
        # The 'lambda' function is used to apply the 'process_row_dep' function to each row.
        # The 'process_row_dep' function is expected to return a dependency tree string for each row,
        # given the row data, the NLP model (self.nlp), and the corpus (self.corpus).
        dep_tree_strs = data.apply(
            lambda row: process_row_dep(row, self.nlp, self.corpus, fit), axis=1
        )

        # The list of dependency tree strings is reshaped into a 2D numpy array,
        # which is required for the following fitting and transformation steps.
        dep_tree_strs_arr = np.array(dep_tree_strs).reshape(-1, 1)

        # If fit=True, fit the one-hot encoder to the dependency tree strings
        if fit:
            # Create a OneHotEncoder object with handle_unknown="ignore"
            # to ignore any categories not seen during fit when transforming data.
            self.dep_encoder = OneHotEncoder(handle_unknown="ignore")
            # Fit the one-hot encoder to the dependency tree strings
            self.dep_encoder.fit(dep_tree_strs_arr)

        # The 'transform' method is used to transform the dependency tree strings
        # into a sparse matrix of one-hot encoded features
        dep_features = self.dep_encoder.transform(dep_tree_strs_arr)

        # Convert the sparse matrix of one-hot encoded features to a DataFrame and return it.
        # The column names are the feature names from the one-hot encoder,
        # and the index is the same as the index of the input data.
        return pd.DataFrame.sparse.from_spmatrix(
            dep_features,
            columns=self.dep_encoder.get_feature_names_out(),
            index=data.index,
        )

    def get_constituency_features(self, data, fit=False):
        """
        This method processes the input data to extract constituency parse strings and encodes them
        into a one-hot encoded format using an OneHotEncoder.

        Parameters:
        data (pandas.DataFrame): The input data to extract constituency parse strings from.
        fit (bool, optional): If True, the one-hot encoder will be fitted to the data.
                            This should be True for the initial training data,
                            and False for any subsequent data. Defaults to False.

        Returns:
        pandas.DataFrame: A DataFrame where each column represents a unique constituency parse string
                        (one-hot encoded) and each row corresponds to the same row in the input data.
        """

        # Apply the process_row_const function to each row of the input data.
        # This will return a Series where each item is a constituency parse string for the corresponding row in the input data.
        constituency_strings = data.apply(
            lambda row: process_row_const(row, self.nlp_benepar), axis=1
        )

        # Reshape the Series to a 2D numpy array so it can be passed to the OneHotEncoder
        constituency_strings_arr = np.array(constituency_strings).reshape(-1, 1)

        # If fit=True, fit the one-hot encoder to the constituency parse strings
        if fit:
            # Create a OneHotEncoder object with handle_unknown="ignore"
            # to ignore any categories not seen during fit when transforming data.
            self.constituency_encoder = OneHotEncoder(handle_unknown="ignore")

            # Fit the one-hot encoder to the constituency parse strings
            self.constituency_encoder.fit(constituency_strings_arr)

        # Use the one-hot encoder to transform the constituency parse strings into a sparse matrix of one-hot encoded features
        constituency_features = self.constituency_encoder.transform(
            constituency_strings_arr
        )

        # Convert the sparse matrix of one-hot encoded features to a DataFrame and return it.
        # The column names are the feature names from the one-hot encoder,
        # and the index is the same as the index of the input data.
        return pd.DataFrame.sparse.from_spmatrix(
            constituency_features,
            columns=self.constituency_encoder.get_feature_names_out(),
            index=data.index,
        )

    def get_sentence_bert_features(self, data, fit=False):
        """
        This method processes the input data to extract sentence embeddings (vectors) for each string
        in the data using Sentence-BERT (en_stsb_distilbert_base) model.

        Parameters:
        data (pandas.DataFrame): The input data to extract sentence vectors from.
        fit (bool, optional): This parameter is not used in this method as Sentence-BERT does not require fitting.
                            However, it's kept for consistency with other similar methods. Defaults to False.

        Returns:
        pandas.DataFrame: A DataFrame where each column represents a dimension of the sentence vector and each
                        row corresponds to the same row in the input data.
        """

        # Apply the sbert_nlp function (which uses Sentence-BERT) to each item in the "string" column of the input data.
        # This will return a Series where each item is a sentence vector for the corresponding string in the input data.
        sentence_vectors = data["string"].apply(lambda x: self.sbert_nlp(x).vector)

        # Convert list of sentence vectors to a numpy array and then stack them vertically.
        # This will create a 2D numpy array where each row is a sentence vector.
        stacked_vectors = np.vstack(sentence_vectors)

        # Convert the 2D numpy array of sentence vectors to a DataFrame and return it.
        # The index is the same as the index of the input data.
        return pd.DataFrame(stacked_vectors, index=data.index)
