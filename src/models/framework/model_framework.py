import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
from .get_modifiers import *
import warnings
import pickle


class PamModel:
    """
    This class represents a predictive model for assigning modifier roles to strings in a given corpus. It utilizes a
    specified machine learning model and feature extraction methodology. The class contains functionality for
    training, saving/loading, prediction, evaluation, and feature importance extraction.

    Attributes:
        model: A Scikit-learn or similar compliant predictive model.
        feature_extractor: An object responsible for transforming the raw data into a form suitable for the model.
        corpus: A Corpus object that contains the full collection of sentences to be used for training and testing.
        train_data: A Pandas DataFrame representing the training portion of the corpus.
        test_data: A Pandas DataFrame representing the testing portion of the corpus.
        X_train: A Pandas DataFrame representing the feature data for the training set.
        y_train: A Pandas Series representing the label data for the training set.
        X_test: A Pandas DataFrame representing the feature data for the testing set.
        y_test: A Pandas Series representing the label data for the testing set.
        predictions: A Numpy array representing the predicted labels for the testing set, as predicted by the model.
    """

    def __init__(self, model, feature_extractor, corpus, train_data, test_data):
        """
        Constructs all the necessary attributes for the PamModel object.

        Args:
            model: A Scikit-learn or similar compliant predictive model.
            feature_extractor: An object responsible for transforming the raw data into a form suitable for the model.
            corpus: A Corpus object that contains the full collection of sentences to be used for training and testing.
            train_data: A Pandas DataFrame or string representing the training portion of the corpus.
            test_data: A Pandas DataFrame or string representing the testing portion of the corpus.
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.corpus = corpus
        self.predictions = None

        # Load training and testing data
        self.train_data = create_modifiers_df(self.corpus, corpus_part=train_data)
        self.test_data = create_modifiers_df(self.corpus, corpus_part=test_data)

        # Separate features and labels
        self.X_train = self.train_data
        self.y_train = self.train_data["role"]
        self.X_test = self.test_data
        self.y_test = self.test_data["role"]

    def train(self):
        """
        This method trains the model on the training data. It extracts features from the training data, fits the model,
        and then generates predictions for the test data.
        """
        # The feature extractor transforms the input data (self.X_train) into a suitable format for the model.
        # The 'fit' parameter indicates whether to fit the feature extractor to the data (i.e., learn the necessary parameters
        # such as the vocabulary for text vectorization).
        # In this case, 'fit' is set to True because for training a model it needs to fit the features to the training data.
        features = self.feature_extractor.transform(self.X_train, fit=True)

        # The model is trained (fit) on the transformed features and the corresponding labels (self.y_train).
        # The 'fit' method of the model learns the parameters of the model that minimize the difference between
        # the model's predictions and the actual labels in the training data.
        self.model.fit(features, self.y_train)

        # After the model has been trained, it is used to generate predictions for the test data.
        # The 'predict' method of this class is used, which transforms the test data using the feature extractor
        # (without fitting it again to the test data) and then generates predictions using the model.
        # These predictions are stored in self.predictions for later use, such as for evaluation.
        self.predictions = self.predict()

    def save_model(self, filename):
        """
        This method is responsible for persisting (saving) the trained model to disk for future use.
        """
        # Open a file in wb. The filename is provided as a parameter to the function.
        with open(filename, "wb") as file:
            # Use the `pickle.dump` function to write the serialized version of the model object (denoted by `self`) to the opened file.
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename):
        """
        This static method is responsible for loading a previously saved model from disk. It is marked as a static method
        because it can be called on the class itself, without needing an instance of the class.
        """
        # Open a file in rb. The filename is provided as a parameter to the function.
        with open(filename, "rb") as file:
            # Use the `pickle.load` function to read the serialized model from the file and de-serialize it
            # (i.e., convert it back into a Python object). The result is a model that is ready to be used for prediction.
            model = pickle.load(file)

        return model

    def predict(self, X=None):
        """
        This method is responsible for generating predictions from the model. It optionally accepts an input data set 'X'.
        If no input data is provided, it defaults to using the stored test data (`self.X_test`).
        """
        # If no input data was provided, use the test data stored in the object.
        if X is None:
            X = self.X_test
        # Transform the input data into the format expected by the model using the feature extractor.
        # Note that 'fit=True' is not passed here because the feature extractor should already be fit to the training data.
        features = self.feature_extractor.transform(X)
        # Generate predictions from the model using the transformed data and return these predictions.
        return self.model.predict(features)

    def evaluate(self, print_report=True):
        """
        This method is responsible for evaluating the performance of the model. It uses the `classification_report` function
        from `sklearn.metrics` to generate a report of the model's performance. The `print_report` argument controls
        whether this report is printed.
        """
        # The classification report is generated using the actual test labels (`self.y_test`) and the predictions
        # that were generated by the model (`self.predictions`). This report includes metrics such as precision,
        # recall, and f1-score that provide insight into the model's performance.
        report = classification_report(self.y_test, self.predictions)
        # If the `print_report` parameter is `True`, print the report to the console. This can be useful for quickly
        # viewing the results without needing to manually inspect the report.
        if print_report:
            print(report)

    def predict_string_from_sentence(self, role_string, sentence_string):
        """
        This method predicts the role of a given string within a given sentence. It takes two arguments, `role_string`,
        which is the string for which we want to predict the role, and `sentence_string`, which is the sentence in
        which the role_string appears. The method then returns a JSON string containing the original string,
        the sentence, and the predicted role.
        """
        # Create a Pandas DataFrame with a single row containing the input `role_string` and `sentence_string`.
        # This DataFrame matches the input format expected by the `predict` method.
        df = pd.DataFrame(
            {"string": [role_string], "sentence_string": [sentence_string]}
        )

        # Use the `predict` method to predict the role of the `role_string` within the `sentence_string`.
        # Note that `self.predict(df)[0]` is used to extract the first (and only) prediction from the result.
        predicted_role = self.predict(df)[0]

        # Create a dictionary that contains the `role_string`, `sentence_string`, and `predicted_role`.
        result = {
            "string": role_string,
            "sentence_string": sentence_string,
            "predicted_role": predicted_role,
        }

        # Convert the dictionary to a JSON string using `json.dumps` and return this string.
        return json.dumps(result)

    def predict_string(self, input_string):
        """
        This method predicts the role of a given string. It takes one argument, `input_string`, which is the string for
        which we want to predict the role. The method then returns a JSON string containing the original string and
        the predicted role.
        """
        # Create a Pandas DataFrame with a single row containing the `input_string`. The DataFrame has a single column,
        # "string", which matches the input format expected by the `predict` method.
        df = pd.DataFrame([input_string], columns=["string"])

        # Use the `predict` method to predict the role of the `input_string`.
        # Note that `self.predict(df)[0]` is used to extract the first (and only) prediction from the result.
        predicted_role = self.predict(df)[0]

        # Create a dictionary that contains the `input_string` and `predicted_role`.
        result = {"string": input_string, "predicted_role": predicted_role}

        # Convert the dictionary to a JSON string using `json.dumps` and return this string.
        return json.dumps(result)

    def evaluate_label(self, label):
        """
        This method evaluates the performance of the model for a specific label. It computes the precision,
        recall, and F1-score for the given label and also provides information on the false positives and
        negatives associated with that label. It returns a dictionary containing these metrics and data.
        """
        # The saved predictions from the model are stored in `self.predictions`.
        predictions = self.predictions

        # It first checks if the provided label exists in the test data labels (`self.y_test`).
        # If the label does not exist, a message is printed and the function returns `None`.
        if label not in self.y_test.unique():
            print(f"Label '{label}' not found in test data.")
            return None

        # To compute the precision, recall, and F1-score for the label, we use `precision_recall_fscore_support`
        # from the `sklearn.metrics` module. A context manager is used to catch and ignore `UndefinedMetricWarning`
        # that could be raised when the label is not present in the predictions.
        with warnings.catch_warnings():  # to handle case when label not in predictions
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            precision, recall, fscore, _ = precision_recall_fscore_support(
                self.y_test, predictions, labels=[label], average="micro"
            )

        # Here the indices of false positives and false negatives are identified for the given label.
        # A false positive is an instance where the model incorrectly predicted the given label.
        # A false negative is an instance where the model failed to predict the given label when it should have.
        fp_index = (self.y_test != label) & (predictions == label)
        fn_index = (self.y_test == label) & (predictions != label)
        false_positives = self.X_test.loc[fp_index].copy()
        false_negatives = self.X_test.loc[fn_index].copy()

        # The predicted labels are then added to the `false_positives` and `false_negatives` dataframes.
        false_positives.loc[:, "predicted_label"] = predictions[fp_index]
        false_negatives.loc[:, "predicted_label"] = predictions[fn_index]

        # A DataFrame is constructed to contain both false positives and false negatives,
        # identified by a new column `false_type` which indicates whether each row is a false positive or negative.
        false_data = pd.concat(
            [
                false_positives.assign(false_type="False Positive"),
                false_negatives.assign(false_type="False Negative"),
            ]
        )
        false_data = false_data[["string", "role", "predicted_label", "false_type"]]

        # Finally, a dictionary is returned containing the precision, recall, F1-score for the given label,
        # and a DataFrame containing the false positives and false negatives for that label.
        return {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
            "false_data": false_data,
        }

    def get_feature_importance(self, label):
        """
        This method is used to extract the importance of each feature for a given label in case the model
        is a Logistic Regression model. If the model is not a Logistic Regression model or the label
        does not exist in the model, a message is printed and the function returns `None`.
        """
        # The `isinstance` function checks if `self.model` is an instance of `LogisticRegression`.
        # If it's not, a message is printed and the function returns `None`.
        if not isinstance(self.model, LogisticRegression):
            print(
                "The model is not Logistic Regression, and feature importance cannot be extracted."
            )
            return None

        # It is then checked if the label exists in the model's classes (`self.model.classes_`).
        # If the label does not exist, a message is printed and the function returns `None`.
        if label not in self.model.classes_:
            print(f"Label '{label}' not found in model classes.")
            return None

        # The coefficients for the label are then extracted from the logistic regression model.
        # These coefficients correspond to the importance of each feature for the given label.
        label_index = list(self.model.classes_).index(label)
        coefficients = self.model.coef_[label_index]

        # The feature names are then gathered from the feature extractor's vectorizer,
        # Vectorizer, POS encoder, dependency encoder, NER encoder, and constituency encoder (if they exist).
        feature_names = []
        if (
            hasattr(self.feature_extractor, "vectorizer")
            and self.feature_extractor.vectorizer is not None
        ):
            feature_names.extend(
                self.feature_extractor.vectorizer.get_feature_names_out()
            )
        if (
            hasattr(self.feature_extractor, "pos_encoder")
            and self.feature_extractor.pos_encoder is not None
        ):
            feature_names.extend(
                self.feature_extractor.pos_encoder.get_feature_names_out()
            )
        if (
            hasattr(self.feature_extractor, "dep_encoder")
            and self.feature_extractor.dep_encoder is not None
        ):
            feature_names.extend(
                self.feature_extractor.dep_encoder.get_feature_names_out()
            )
        if (
            hasattr(self.feature_extractor, "ner_encoder")
            and self.feature_extractor.ner_encoder is not None
        ):
            feature_names.extend(
                self.feature_extractor.ner_encoder.get_feature_names_out()
            )
        if (
            hasattr(self.feature_extractor, "constituency_encoder")
            and self.feature_extractor.constituency_encoder is not None
        ):
            feature_names.extend(
                self.feature_extractor.constituency_encoder.get_feature_names_out()
            )

        # A DataFrame is created to store the feature names and their corresponding coefficients (importance).
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": coefficients}
        )

        # The DataFrame is then sorted by the importance of the features in descending order.
        feature_importance_df = feature_importance_df.sort_values(
            by="importance", ascending=False
        )

        # The sorted DataFrame is returned by the function.
        return feature_importance_df
