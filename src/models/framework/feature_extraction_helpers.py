# feature_extractor_helpers.py

import re
from spacy.tokens import Span
from nltk.tree import Tree
import pandas as pd

UNPROCESSED = "unprocessed"


def normalize_text(text):
    """
    This function normalizes the input text by removing any punctuation and converting all characters to lowercase.

    Parameters:
    text (str): The input string that needs to be normalized.

    Returns:
    str: The normalized string with all characters in lowercase and no punctuation.
    """

    # Use a regular expression to replace anything that is not a word character or whitespace with nothing,
    # effectively removing all punctuation. Then convert the resulting string to lowercase.
    return re.sub(r"[^\w\s]", "", text.lower())


def strip_tokens(tree):
    """
    This function recursively strips the actual tokens from a parse tree, replacing them with the string "TOKEN".
    It's used to anonymize the tree, preserving only the tree structure and the POS tags.

    Parameters:
    tree (nltk.tree.Tree or str): A parse tree or a leaf node string from a parse tree.

    Returns:
    nltk.tree.Tree or str: The anonymized parse tree or leaf node string.
    """

    # Check if the input is a leaf node (a string).
    if isinstance(tree, str):
        # If it is, simply return the string "TOKEN".
        return "TOKEN"
    else:
        # If it's not a leaf node (it's a subtree), then recursively apply this function to all its children.
        # The label of the subtree is preserved.
        return Tree(tree.label(), [strip_tokens(child) for child in tree])


def find_role_span_const(row, nlp_benepar):
    """
    This function finds the span of a role in a sentence and returns the span, the sentence's doc, and the role's doc.

    Parameters:
    row (pandas.Series): A row from a DataFrame that contains the role string and the sentence string.
    nlp_benepar (spacy.language.Language): The SpaCy Language model with Benepar parser.

    Returns:
    tuple: A tuple containing the role span, sentence doc, and role doc, or None if the role is not found in the sentence.
    """

    # Get the role and sentence strings from the row.
    role_string = row["string"]
    sentence_string = row["sentence_string"]

    # Parse the sentence string using the NLP model to get a Doc object.
    sentence_doc = nlp_benepar(sentence_string)

    # Extract the tokens from the sentence Doc object.
    tokens = [token.text for token in sentence_doc]

    # Parse the role string using the NLP model to get another Doc object.
    role_doc = nlp_benepar(role_string)

    # Extract the tokens from the role Doc object.
    role_tokens = [token.text for token in role_doc]

    # Loop over the sentence tokens.
    for i in range(len(tokens)):
        # Check if the tokens at the current position match the role tokens.
        if tokens[i : i + len(role_tokens)] == role_tokens:
            # If they do, calculate the start and end positions of the role span.
            start = i
            end = i + len(role_tokens)

            # Create a Span object for the role span.
            role_span = Span(sentence_doc, start, end)

            # Return the role span, sentence doc, and role doc.
            return role_span, sentence_doc, role_doc

    # If the function hasn't returned yet, it means the role wasn't found in the sentence. In this case, return None.
    return None


def process_row_const(row, nlp_benepar):
    """
    This function processes a row of a DataFrame to extract a constituency parse tree of a role span in a sentence.
    It normalizes the text and strips the actual tokens from the parse tree, preserving only the tree structure
    and the POS tags.

    Parameters:
    row (pandas.Series): A row from a DataFrame that contains the role string and the sentence string.
    nlp_benepar (spacy.language.Language): The SpaCy Language model with Benepar parser.

    Returns:
    str: A string representation of the stripped constituency parse tree of the role span, or "NO_MATCH" if the role
    span could not be generated or if no sentence spans were found meeting the specified condition.
    """

    # Call the find_role_span_const function to find the role span, sentence doc, and role doc.
    result = find_role_span_const(row, nlp_benepar)

    # If the function returned None, it means the role span could not be generated.
    if result is None:
        print("Failed to generate role span. Check the start and end indices.")
        parse_subtree_stripped = Tree("NO_MATCH", [])
        return str(parse_subtree_stripped)

    # If the function returned a result, unpack it into the role span, sentence doc, and role doc.
    role_span, sentence_doc, role_doc = result

    # If a role span was generated, process it further.
    if role_span is not None:
        # Find all sentence spans that contain the entire role span.
        sentence_spans = [
            sent
            for sent in sentence_doc.sents
            if role_span.start >= sent.start and role_span.end <= sent.end
        ]

        # If there are any sentence spans that meet the condition, process the first one.
        if sentence_spans:  # Check if sentence_spans is not empty
            sentence_span = sentence_spans[0]

            # Parse the sentence span into a parse tree.
            parse_tree = Tree.fromstring(sentence_span._.parse_string)

            # Normalize the role string by removing punctuation and converting to lowercase.
            normalized_role_string = normalize_text(role_doc.text)

            # Find the subtree of the parse tree that corresponds to the role.
            for subtree in parse_tree.subtrees():
                # Normalize the text of the subtree and compare it to the normalized role string.
                if normalize_text(" ".join(subtree.leaves())) == normalized_role_string:
                    parse_subtree = subtree
                    break
            else:
                # If no suitable subtree was found in the parse tree, generate a new parse tree from the role doc.
                role_span_const = [sent for sent in role_doc.sents][0]
                parse_tree = Tree.fromstring(role_span_const._.parse_string)
                parse_subtree = parse_tree

            # Strip the actual tokens from the parse subtree, preserving only the tree structure and the POS tags.
            parse_subtree_stripped = strip_tokens(parse_subtree)
        else:
            # If no sentence spans met the condition, print a message and set the stripped parse subtree to "NO_MATCH".
            print("No sentence spans found meeting the specified condition.")
            parse_subtree_stripped = Tree("NO_MATCH", [])
    else:
        # If a role span could not be generated, print a message and set the stripped parse subtree to "NO_MATCH".
        print("Failed to generate role span. Check the start and end indices.")
        parse_subtree_stripped = Tree("NO_MATCH", [])

    # Return the string representation of the stripped parse subtree.
    return str(parse_subtree_stripped)


def find_role_span_dep(row, nlp, corpus, fit):
    """
    This function locates the position of a specified role string within a text document. It returns a Span object
    which represents the start and end tokens of the role string within the text. If the role string is not found
    within the text, it returns None.

    Parameters:
    row (pandas.Series): A row from a DataFrame that contains the role string and the sentence string.
    nlp (spacy.language.Language): The SpaCy Language model.
    corpus (Corpus): The Corpus object from which we can retrieve role and sentence information by ID.
    fit (bool): Whether we are in the fitting phase or not.

    Returns:
    spacy.tokens.Span: A Span object representing the role_string in doc. If role_string is not found, return None.
    """

    # If in the fitting phase, get the role and sentence information from the corpus.
    if fit:
        role = corpus.get_role_by_id(row["role_id"])
        sentence = corpus.get_sentence(role.sentence_id)
        sentence_string = sentence.sentence_string
        role_string = role.string
        doc = sentence.doc
    else:
        # If not in the fitting phase, get the role and sentence information from the row.
        role_string = row["string"]
        sentence_string = row["sentence_string"]
        doc = nlp(sentence_string)

    # Get a list of all tokens in the document.
    tokens = [token.text for token in doc]
    # Split the role string into individual tokens.
    role_tokens = role_string.split()

    # First method: Try to find the role tokens in the document tokens by splitting the role_string on spaces.
    for i in range(len(tokens)):
        if tokens[i : i + len(role_tokens)] == role_tokens:
            start = i
            end = i + len(role_tokens)  # end index in Span should be exclusive
            role_span = Span(doc, start, end)
            return role_span

    # Second method: Create a Doc from role_string. Used as a fallback if the first method fails.
    role_doc = nlp(role_string)
    role_tokens = [token.text for token in role_doc]
    for i in range(len(tokens)):
        if tokens[i : i + len(role_tokens)] == role_tokens:
            start = i
            end = i + len(role_tokens)  # end index in Span should be exclusive
            role_span = Span(doc, start, end)
            return role_span

    # If role string is not found in the document, return None.
    return None


def extract_dep_labels(token):
    """
    This function extracts the dependency labels of a given token along with its children tokens in the parse tree.
    It recursively visits each child of the given token and forms a string representation of their dependency labels.

    Parameters:
    token (spacy.tokens.Token): The token from a parse tree for which we want to extract dependency labels.

    Returns:
    str: A string representing the token's dependency label followed by the dependency labels of its children.
    """

    # Calculate the number of children tokens the input token has by adding the number of left and right children.
    # n_lefts and n_rights attributes give the number of syntactic children a token has.
    has_children = token.n_lefts + token.n_rights > 0

    # If the token has child/children tokens
    if has_children:
        # Recursively call this function for each child token and join all results into a string separated by spaces.
        # This gives us the dependency labels of all child tokens in the format "(dep_label child_dep_labels)".
        children_labels = " ".join(
            [extract_dep_labels(child) for child in token.children]
        )
    else:
        # If the token does not have any child tokens, set the children labels string as empty.
        children_labels = ""

    # Concatenate the token's own dependency label with its children's labels.
    # Each dependency label is enclosed in brackets. The token's dependency label comes first,
    # followed by a space and then the children's labels.
    dep_labels = f"({token.dep_} {children_labels})"

    # Return the final string.
    return dep_labels


def process_row_dep(row, nlp, corpus, fit):
    """
    This function extracts dependency labels from the role span of the row's 'role_id' or 'string' in the DataFrame.
    If 'role_id' or 'string' are missing or NaN, the function returns a predefined UNPROCESSED value.

    Parameters:
    row (pd.Series): A row from a DataFrame which includes 'role_id' or 'string' values.
    nlp (spacy.lang model): A SpaCy language model to process the strings into Docs.
    corpus (object): A corpus object that has a method get_role_by_id.
    fit (bool): A boolean that represents whether we are in the fitting phase or not.

    Returns:
    str: A string of the root token's dependency label in the role span, and its children. If no valid span, return UNPROCESSED.
    """
    # Initialising role_span as None. This will hold the Span object of the role string in the document if found.
    role_span = None

    # Check if 'role_id' is present in the row and is not NaN
    if "role_id" in row and pd.notna(row["role_id"]):
        # Call the find_role_span_dep function with the row's 'role_id', nlp model, corpus and fit parameter
        # to get the Span object for the role string in the document.
        role_span = find_role_span_dep(row, nlp, corpus, fit)
    # Else, check if both 'string' and 'sentence_string' are present in the row and neither is NaN.
    elif all(
        key in row and pd.notna(row[key]) for key in ["string", "sentence_string"]
    ):
        # If both are present and not NaN, call the find_role_span_dep function with the row's 'string' and 'sentence_string',
        # nlp model, corpus and fit parameter to get the Span object for the role string in the document.
        role_span = find_role_span_dep(row, nlp, corpus, fit)

    # If a valid role span is found
    if role_span is not None:
        # Select tokens from the role span such that the head (parent) of the token is not in the role span.
        # This gives us the root(s) of the span, which is the token(s) at the top of the dependency subtree within the span.
        tokens = [token for token in role_span if token.head not in role_span]
        # If there are any root tokens
        if tokens:
            # Choose the first root token.
            root = tokens[0]
            # Extract dependency labels from the root token's subtree and return the result.
            return extract_dep_labels(root)

    # If 'role_id' is not in row or it is NaN, or if no root tokens were found, return the predefined UNPROCESSED value.
    return UNPROCESSED
