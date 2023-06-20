import pandas as pd
import os  # nopep8
import sys  # nopep8
from pathlib import Path  # nopep8

# Add the 'src' directory to sys.path
current_working_directory = os.getcwd()  # Get the current working directory
src_path = (
    Path(current_working_directory).resolve() / "src"
)  # Construct the path to the 'src' directory
sys.path.append(str(src_path))  # Add 'src' directory to sys.path

from corpus import *


def create_roles_df(corpus, corpus_part="train"):
    """
    This function creates a DataFrame of roles present in the provided corpus. Each role's details are represented
    as a separate row in the DataFrame.

    Parameters:
    corpus (object): The corpus object from which to extract roles. It should have attributes 'train', 'dev', 'test' and 'sentences' representing different parts of the corpus.
    corpus_part (str, optional): The part of the corpus to use for creating the DataFrame. Default is 'train'.

    Returns:
    pandas.DataFrame: A DataFrame where each row represents a role's details from the corpus.
    """

    # Initialize an empty list to store the role details as dictionary objects, where each dictionary will correspond to a row in the DataFrame
    data = []

    # Depending on the specified corpus_part, choose the appropriate sentences collection from the corpus.
    if corpus_part == "train":
        sentences = corpus.train
    elif corpus_part == "dev":
        sentences = corpus.dev
    elif corpus_part == "test":
        sentences = corpus.test
    elif corpus_part == "full":
        sentences = corpus.sentences
    else:
        print("Invalid corpus_part specified. Defaulting to 'train'")
        sentences = corpus.train

    # Iterate over all sentences in the chosen sentences collection
    for sentence_id, sentence in sentences.items():
        # For each sentence, iterate over all its frames.
        for frame in sentence.frames:
            # For each frame, iterate over all its roles.
            for role in frame.roles:
                # For each role, append a dictionary to the data list. This dictionary contains the role's details,
                # and each key-value pair will correspond to a column-value pair in the final DataFrame.
                data.append(
                    {
                        "sentence_id": sentence_id,
                        "sentence_string": sentence.sentence_string,
                        "frame": frame.frame_name,
                        "role": role.role,
                        "role_id": role.role_id,
                        "indices": role.indices,
                        "string": role.string,
                    }
                )

    # Convert the data list (a list of dictionaries) into a pandas DataFrame.
    df = pd.DataFrame(data)

    # Define the set of core roles. These are the standard roles that are common across different frames.
    core_roles = {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5", "ARGA", "V"}

    # Add a new column 'role_type' to the DataFrame, which specifies whether each role is a core role or a modifier role.
    # This is done by applying a lambda function on the 'role' column, which maps each role to its category based on whether it is in the core_roles set.
    df["role_type"] = df["role"].apply(
        lambda x: "core_role" if x in core_roles else "modifier_role"
    )

    # Return the final DataFrame
    return df


def create_modifiers_df(corpus, corpus_part="train"):
    """
    This function creates a DataFrame of modifier roles present in the provided corpus. Each modifier role's details are represented
    as a separate row in the DataFrame. Only modifier roles which account for more than 5% of the total modifier roles are included.

    Parameters:
    corpus (object): The corpus object from which to extract roles. It should have attributes 'train', 'dev', 'test' and 'sentences' representing different parts of the corpus.
    corpus_part (str, optional): The part of the corpus to use for creating the DataFrame. Default is 'train'.

    Returns:
    pandas.DataFrame: A DataFrame where each row represents a modifier role's details from the corpus.
    """

    # Call create_roles_df() function to create a DataFrame with all roles in the specified part of the corpus.
    df = create_roles_df(corpus, corpus_part)

    # Filter the DataFrame to only include rows where the role type is 'modifier_role'.
    modifiers = df[df["role_type"] == "modifier_role"]

    # Calculate the relative frequency (i.e., percentage of total modifier roles) of each unique modifier role.
    role_counts = modifiers["role"].value_counts(normalize=True)

    # Create a series where each value indicates whether the corresponding role's frequency is more than 5%.
    frequent_roles = role_counts > 0.05

    # Use indexing to filter the 'modifiers' DataFrame to only include rows where the role is in the frequent_roles index.
    # This effectively filters the DataFrame to only include rows for modifier roles that account for more than 5% of the total modifier roles.
    modifiers = modifiers[modifiers["role"].isin(frequent_roles.index[frequent_roles])]

    # Return the filtered DataFrame
    return modifiers
