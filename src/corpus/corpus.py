from typing import Dict
import re
from tabulate import tabulate
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import spacy, benepar
from spacy import displacy
from spacy.tokens import Doc, Span

# from spacy.symbols import ORTH
from IPython.display import display, HTML
import pandas as pd
import pickle
import csv
from tqdm import tqdm


@dataclass
class Role:
    """
    Data class for a 'Role'. This role represents a single argument of a frame. This argument can be a core role (ARG0, ARG1, ...) or a modifier role (TMP, LOC, MNR, ...).

    Attributes:
        sentence_id (Tuple[str, int]): Unique identifier for the sentence where this role exists.
        role_id (int): Unique identifier for this role.
        role (str): String representation of this role.
        indices (list): List of indices in the sentence where the role occurs.
        string (str): The actual text of the role in the sentence.
        role_index (int): Index of the role within its containing sentence.
        va_role (str, optional): VerbAtlas role equivalent for this role. Defaults to None.
        doc (spacy.tokens.Doc, optional): Spacy Doc object for the role. Defaults to None.
    """

    sentence_id: Tuple[str, int]
    role_id: int
    role: str
    indices: list
    string: str
    role_index: int
    va_role: Optional[str] = None
    doc: Optional[Doc] = None


@dataclass
class Frame:
    """
    Data class for a 'Frame'. A frame consists of Frame Evoking Element (often a verb) and a set of roles that are associated with it.

    Attributes:
        frame_name (str): Name of the frame.
        lemma_name (str): Lemma or base form of the verb in the frame.
        roles (List[Role]): List of roles that are part of this frame.
        va_frame_id (str, optional): VerbAtlas frame id equivalent for this frame. Defaults to None.
        va_frame_name (str, optional): VerbAtlas frame name equivalent for this frame. Defaults to None.
    """

    frame_name: str
    lemma_name: str
    roles: List[Role]
    va_frame_id: Optional[str] = None
    va_frame_name: Optional[str] = None


@dataclass
class Sentence:
    """
    Data class for a 'Sentence'. A sentence can contain multiple frames.

    Attributes:
        sentence_id (Tuple[str, int]): Unique identifier for the sentence.
        sentence_string (str): The actual text of the sentence.
        frames (List[Frame]): List of frames that are part of this sentence.
        doc (spacy.tokens.Doc): Spacy Doc object for the sentence.
    """

    sentence_id: Tuple[str, int]
    sentence_string: str
    frames: List[Frame]
    doc: Doc = field(default=None, init=False)


class VerbAtlasFrames:
    """
    Class for handling VerbAtlas frames. This class contains a mapping between PropBank senses and VerbAtlas frames.

    Attributes:
        mapping (dict): Dictionary mapping between PropBank senses and VerbAtlas frames.
        frame_info (dict): Information about VerbAtlas frames.
    """

    def __init__(self, mapping_file_path: str, frame_info_file_path: str):
        """
        Initializes a VerbAtlasFrames object.

        Args:
            mapping_file_path (str): Path to the file containing PropBank to VerbAtlas mapping.
            frame_info_file_path (str): Path to the file containing information about VerbAtlas frames.
        """
        self.mapping = self._load_mapping(mapping_file_path)
        self.frame_info = self._load_frame_info(frame_info_file_path)

    def _load_mapping(self, mapping_file_path):
        """
        Load a mapping file where each line contains a mapping from a PropBank frame to a VerbAtlas frame.
        Each line is formatted as follows: 'pb_sense>va_frame_id    pb_role>va_role    ...'

        :param mapping_file_path: The path to the mapping file.
        :return: A dictionary where the key is the PropBank frame and the value is another dictionary
        containing the VerbAtlas frame ID and the role mappings.
        """

        # Initialize an empty dictionary to hold the mappings
        mapping = {}

        # Open the file at the provided file path
        with open(mapping_file_path) as csvfile:
            # Create a CSV reader object to read the file line by line, separating each line at the tab character
            csv_reader = csv.reader(csvfile, delimiter="\t")

            # Iterate through each row of the CSV file
            for row in csv_reader:
                # If the row is empty or the first cell of the row does not contain a ">" character, skip this row
                if not row or ">" not in row[0]:
                    continue

                # Split the first cell of the row at the ">" character to get the PropBank frame sense and VerbAtlas frame ID
                pb_sense, va_frame_id = row[0].split(">")

                # Initialize an empty dictionary to hold the role mappings for this frame sense
                role_mappings = {}

                # Iterate over the rest of the cells in the row, each of which contains a role mapping
                for role_map in row[1:]:
                    # Split the cell at the ">" character to get the PropBank role and VerbAtlas role
                    pb_role, va_role = role_map.split(">")

                    # Convert the PropBank role to a normalized form
                    converted_pb_role = self._convert_propbank_role(pb_role)

                    # Add the role mapping to the dictionary, using the normalized PropBank role as the key and the VerbAtlas role as the value
                    role_mappings[converted_pb_role] = va_role

                # Add the frame sense mapping to the main dictionary, using the PropBank frame sense as the key and a dictionary
                # containing the VerbAtlas frame ID and the role mappings as the value
                mapping[pb_sense] = {
                    "va_frame_id": va_frame_id,
                    "role_mappings": role_mappings,
                }

        # Return the dictionary containing all the mappings
        return mapping

    def _load_frame_info(self, frame_info_file_path):
        """
        Load a frame information file where each line contains the ID of a VerbAtlas frame and some associated information.
        Each line is formatted as follows: 'frame_id    frame_info'

        :param frame_info_file_path: The path to the frame information file.
        :return: A dictionary where the key is the VerbAtlas frame ID and the value is the associated frame information.
        """

        # Initialize an empty dictionary to hold the frame information
        frame_info = {}

        # Open the file at the provided file path
        with open(frame_info_file_path) as csvfile:
            # Create a CSV reader object to read the file line by line, separating each line at the tab character
            csv_reader = csv.reader(csvfile, delimiter="\t")

            # Iterate through each row of the CSV file
            for row in csv_reader:
                try:
                    # Attempt to extract the frame ID from the first cell of the row
                    frame_id = row[0]

                    # If the frame ID does not start with "va:" or the row has fewer than two cells, skip this row
                    if not frame_id.startswith("va:") or len(row) < 2:
                        continue

                    # Attempt to extract the frame number from the frame ID by removing the first three characters and the last character,
                    # and converting the remaining string to an integer
                    frame_number = int(frame_id[3:-1])

                    # Add the frame information to the dictionary, using the frame ID as the key and the information in the second cell as the value
                    frame_info[frame_id] = row[1]

                # If a ValueError or IndexError is raised during the processing of a row (for example, if the frame ID does not contain
                # a valid number, or if the row has fewer than two cells), skip this row and continue with the next one
                except (ValueError, IndexError):
                    continue

        # Return the dictionary containing all the frame information
        return frame_info

    def get_verbatlas_mapping(self, propbank_sense: str) -> Dict[str, str]:
        """
        Retrieve the VerbAtlas frame mapping for a given PropBank sense.

        :param propbank_sense: The PropBank sense for which to retrieve the mapping.
        :return: A dictionary containing the VerbAtlas frame ID, name, and role mappings corresponding to the provided PropBank sense.
        """

        # Convert the PropBank sense to lowercase and attempt to retrieve the corresponding VerbAtlas mapping from the instance's mapping dictionary.
        # If no such mapping exists, retrieve an empty dictionary instead.
        mapping = self.mapping.get(propbank_sense.lower(), {})

        # From the retrieved mapping, attempt to extract the VerbAtlas frame ID. If no such frame ID exists, use "unknown" as a default value.
        va_frame_id = mapping.get("va_frame_id", "unknown")

        # Using the retrieved VerbAtlas frame ID, attempt to find the corresponding frame name in the instance's frame info dictionary.
        # If no such frame name exists, use "unknown" as a default value.
        va_frame_name = self.frame_info.get(va_frame_id, "unknown")

        # From the retrieved mapping, attempt to extract the role mappings. If no such mappings exist, retrieve an empty dictionary instead.
        role_mappings = mapping.get("role_mappings", {})

        # Add the PropBank role "V" to the VerbAtlas role mappings, mapping it to "V" in VerbAtlas.
        role_mappings["V"] = "V"

        # Add the PropBank role "FEE" to the VerbAtlas role mappings, mapping it to "FEE" in VerbAtlas.
        role_mappings["FEE"] = "FEE"

        # Return a dictionary containing the VerbAtlas frame ID, name, and role mappings.
        return {
            "va_frame_id": va_frame_id,
            "va_frame_name": va_frame_name,
            "role_mappings": role_mappings,
        }

    def _convert_propbank_role(self, propbank_role: str) -> str:
        """
        Convert PropBank roles to a simplified form.

        :param propbank_role: The original PropBank role as a string.
        :return: A string representing the simplified role.
        """

        # Use a regular expression to match roles that start with "ARG" followed by a digit.
        # The r before the string indicates a raw string, which allows backslashes to be interpreted as literal backslashes.
        arg_match = re.match(r"ARG(\d)", propbank_role)
        if arg_match:
            # If a match is found, replace "ARG" with "A" and return the new string.
            # The group(1) method returns the first parenthesized subgroup in the match, which in this case is the digit following "ARG".
            return f"A{arg_match.group(1)}"

        # Use a regular expression to match roles that start with "AM-" followed by one or more word characters.
        am_match = re.match(r"AM-(\w+)", propbank_role)
        if am_match:
            # If a match is found, remove "AM-" and return the remaining string.
            # The group(1) method returns the first parenthesized subgroup in the match, which in this case is the string following "AM-".
            return am_match.group(1)

        # If the PropBank role does not match either of the above formats, return it unmodified.
        return propbank_role


class Corpus:
    """
    Corpus Class

    The Corpus class facilitates the loading and processing of text data. The class takes file paths for training,
    development, and testing datasets and uses these datasets to initialize a series of attributes that contain information
    about sentences, roles, and verb atlas frames. The sentences are processed and stored in various forms and are mapped
    to their respective roles. The class also provides methods to access the processed sentences, roles, and verb atlas frames.

    Attributes:
    sentences (Dict[Tuple[int, int], Sentence]): Dictionary mapping unique sentence IDs to Sentence objects.
    roles (Dict[int, Role]): Dictionary mapping unique role IDs to Role objects.
    train (Dict[Tuple[int, int], Sentence]): Dictionary mapping unique sentence IDs to Sentence objects in the training data.
    dev (Dict[Tuple[int, int], Sentence]): Dictionary mapping unique sentence IDs to Sentence objects in the development data.
    test (Dict[Tuple[int, int], Sentence]): Dictionary mapping unique sentence IDs to Sentence objects in the test data.
    source_file_index_map (Dict[str, int]): Dictionary mapping source file names to unique index.
    verb_atlas_frames (VerbAtlasFrames): An instance of VerbAtlasFrames containing mappings for verb atlas.
    role_id_map (Dict[int, Role]): Dictionary mapping unique role IDs to Role objects.
    nlp (spacy.lang.en.English): An instance of SpaCy's English model.
    raw_sentences (List[str]): List of raw sentence strings.
    raw_role_strings (List[str]): List of raw role strings.
    sentence_id_mapping (Dict[int, Tuple[int, int]]): Dictionary mapping an index to a unique sentence ID.
    role_id_mapping (Dict[int, int]): Dictionary mapping an index to a unique role ID.
    """

    def __init__(
        self,
        train_file_path: str,
        dev_file_path: str,
        test_file_path: str,
        verb_atlas_frames: VerbAtlasFrames,
        data_ratio: float = 1.0,
    ):
        self.sentences: Dict[Tuple[int, int], Sentence] = {}
        self.roles: Dict[int, Role] = {}
        self.train: Dict[Tuple[int, int], Sentence] = {}
        self.dev: Dict[Tuple[int, int], Sentence] = {}
        self.test: Dict[Tuple[int, int], Sentence] = {}
        self.source_file_index_map: Dict[str, int] = {}
        self.verb_atlas_frames = verb_atlas_frames
        self.role_id_map = {}

        # initialize the SpaCy model
        self.nlp = spacy.load("en_core_web_sm")
        # self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

        self.raw_sentences = []
        self.raw_role_strings = []
        self.sentence_id_mapping = {}
        self.role_id_mapping = {}

        try:
            self._load_corpus_data(train_file_path, "train", data_ratio)
            self._process_spacy_data(self.train, "train")

            self._load_corpus_data(dev_file_path, "dev", data_ratio)
            self._process_spacy_data(self.dev, "dev")

            self._load_corpus_data(test_file_path, "test", data_ratio)
            self._process_spacy_data(self.test, "test")

        except UnicodeDecodeError as e:
            print(f"Error decoding file: {e}")
            return

    def _process_spacy_data(self, data, name):
        """
        Process raw sentences using SpaCy and associate them with their unique identifiers.

        :param data: A dictionary where keys are unique sentence identifiers and values are instances of a Sentence class.
        :param name: A string representing the name of the data being processed. This is used for printing progress updates.
        """

        # Start processing sentences with SpaCy. Print a message indicating the start of this process.
        print(f"Processing {name} sentences with SpaCy...")

        # Use SpaCy's pipeline to process each sentence in self.raw_sentences. The tqdm function is used to create a progress bar.
        sentence_docs = list(
            self.nlp.pipe(tqdm(self.raw_sentences, desc=f"Processing {name} sentences"))
        )

        # Process raw role strings with SpaCy if uncommented.
        # print(f"Processing {name} roles with SpaCy...")
        # role_docs = list(
        #     self.nlp.pipe(tqdm(self.raw_role_strings, desc=f"Processing {name} roles"))
        # )

        # Print a message indicating the start of setting sentence docs in the data dictionary.
        print(f"Setting {name} sentence docs in the right place...")

        # Loop through each processed sentence doc and its index.
        for idx, doc in tqdm(
            enumerate(sentence_docs),
            desc=f"Setting {name} sentence docs",
            total=len(sentence_docs),
        ):
            # Retrieve the unique sentence id associated with the current index.
            unique_sentence_id = self.sentence_id_mapping[idx]

            # Assign the processed sentence doc to the appropriate place in the data dictionary, using the unique sentence id as the key.
            data[unique_sentence_id].doc = doc

            # print(f"Setting {name} role docs in the right place...")
            # for idx, doc in tqdm(
            #     enumerate(role_docs),
            #     desc=f"Setting {name} role docs",
            #     total=len(role_docs),
            # ):
            #     unique_role_id = self.role_id_mapping[idx]
            #     if doc is None:  # reprocess the role string with SpaCy if doc is None
            #         print(
            #             f"Warning: Encountered None doc for role_id {unique_role_id}, reprocessing with SpaCy..."
            #         )
            #         self.roles[unique_role_id].doc = self.nlp(
            #             self.roles[unique_role_id].string
            #         )
            #     else:
            #         self.roles[unique_role_id].doc = doc
            # print("All roles processed successfully.")

    def _parse_frame_roles(
        self, frame_roles: str, sentence_id: Tuple[int, int]
    ) -> List[Role]:
        """
        Parse frame roles from a given string and return a list of Role objects.

        :param frame_roles: A string containing all the frame roles, in a specific format.
        :param sentence_id: A tuple of two integers representing the unique sentence identifier.
        :return: A list of Role objects parsed from the frame_roles string.
        """

        # Initialize an empty list to store the Role objects.
        roles = []

        # Define the regular expression pattern for matching frame roles in the input string.
        role_pattern = r"(\w+) \[(.+?)\]: (.+?)(?=;|$)"

        # Use the finditer function to match the pattern in the frame_roles string.
        for i, match in enumerate(re.finditer(role_pattern, frame_roles)):
            # Extract the role, indices and string from the match groups.
            role, indices, string = match.groups()

            # Convert the string of indices into a list of integers.
            indices = list(map(int, indices.split(" | ")))

            # Replace "~" with "," in the string.
            string = string.replace("~", ",")

            # Generate a unique role id by concatenating sentence id and the match index, and convert it to an integer.
            role_id = int(f"{sentence_id[0]}{sentence_id[1]}{i}")

            # Get the current index in the roles list. This will be the index of the new Role object in the roles list.
            role_index = len(roles)

            # Create a new Role object with the parsed data.
            role_instance = Role(
                role_id=role_id,
                role=role,
                indices=indices,
                string=string,
                role_index=role_index,
                sentence_id=sentence_id,
            )

            # Add the Role object to the roles list.
            roles.append(role_instance)

            # Add the Role object to the role_id_map dictionary with the role id as the key.
            self.role_id_map[role_id] = role_instance

        # Return the list of Role objects.
        return roles

    def get_sentences(self) -> Dict[Tuple[int, int], str]:
        """
        Get the sentence strings from all Sentence objects in the class instance.

        :return: A dictionary where the keys are sentence ids and the values are sentence strings.
        """

        # Initialize an empty dictionary to store the sentence ids and corresponding sentence strings.
        sentence_dict = {}

        # Loop over the items in the sentences dictionary of the class instance. The keys are sentence ids
        # and the values are Sentence objects.
        for sentence_id, sentence in self.sentences.items():
            # Add the sentence id and the sentence string of the Sentence object to the sentence_dict.
            # The sentence string is a property of the Sentence object.
            sentence_dict[sentence_id] = sentence.sentence_string

        # Return the dictionary of sentence ids and corresponding sentence strings.
        return sentence_dict

    def get_gold_standard_frames(
        self, sentence_id: Tuple[int, int]
    ) -> Optional[List[Dict[str, any]]]:
        """
        Get the gold standard frames for a given sentence id.

        :param sentence_id: The id of the sentence for which the gold standard frames are required.
        :return: A list of dictionaries where each dictionary represents a frame, if the sentence_id is found.
                Returns None if the sentence_id is not found.
        """

        # Check if the sentence_id exists in the sentences dictionary of the class instance.
        # The keys of the sentences dictionary are sentence ids.
        if sentence_id in self.sentences:
            # If the sentence_id is found, return the frames of the corresponding Sentence object.
            # The frames is a property of the Sentence object.
            return self.sentences[sentence_id].frames

        # If the sentence_id is not found in the sentences dictionary, return None.
        return None

    def add_verbatlas_mappings(self):
        """
        Add Verbatlas mapping information to the frames and roles of each sentence in the dataset.
        """

        # Loop over each item in the sentences dictionary.
        # The key (sentence_id) represents the unique identifier for a sentence,
        # and the value (sentence) is the Sentence object associated with that id.
        for sentence_id, sentence in self.sentences.items():
            # Loop over each Frame object in the frames list of the Sentence object.
            for frame in sentence.frames:
                # The frame_name property of a Frame object represents the PropBank id for that frame.
                # Store it in the propbank_id variable.
                propbank_id = frame.frame_name

                # Use the get_verbatlas_mapping method of the verb_atlas_frames object to get the
                # Verbatlas mapping for the PropBank id stored in propbank_id.
                # Store the returned mapping dictionary in the mapping variable.
                mapping = self.verb_atlas_frames.get_verbatlas_mapping(propbank_id)

                # Update the va_frame_id and va_frame_name properties of the Frame object using the
                # corresponding values from the mapping dictionary.
                frame.va_frame_id = mapping["va_frame_id"]
                frame.va_frame_name = mapping["va_frame_name"]

                # Loop over each Role object in the roles list of the Frame object.
                for role in frame.roles:
                    # The role property of a Role object represents the PropBank role.
                    # Store it in the pb_role variable.
                    pb_role = role.role

                    # Use the _convert_propbank_role method of the verb_atlas_frames object to convert the
                    # PropBank role stored in pb_role to a format suitable for Verbatlas.
                    # Then, use the converted role to get the corresponding Verbatlas role from the
                    # role_mappings dictionary in the mapping variable.
                    # If there is no matching role, default to "unknown".
                    va_role = mapping["role_mappings"].get(
                        self.verb_atlas_frames._convert_propbank_role(pb_role),
                        "unknown",
                    )

                    # Update the va_role property of the Role object with the value stored in va_role.
                    role.va_role = va_role

    def _source_file_to_index(self, source_file: str) -> int:
        """
        Convert a source file name to a unique index.
        """

        # Check if the source_file string is not in the keys of the source_file_index_map dictionary.
        # The source_file_index_map dictionary stores the unique index for each source file name.
        if source_file not in self.source_file_index_map:
            # If the source_file string is not in the dictionary, create a new index for it.
            # The index is the current length of the source_file_index_map dictionary.
            index = len(self.source_file_index_map)

            # Add a new key-value pair to the source_file_index_map dictionary.
            # The key is the source_file string and the value is the new index.
            self.source_file_index_map[source_file] = index

        # Return the index associated with the source_file string from the source_file_index_map dictionary.
        # If the source_file string was not in the dictionary before this method was called,
        # this returns the new index created above. Otherwise, it returns the existing index.
        return self.source_file_index_map[source_file]

    def _load_corpus_data(self, file_path: str, split: str, data_ratio: float):
        """
        Loads data from a given file, modifies it, and stores it in memory.
        """
        # Initialize lists for storing raw sentences and role strings, as well as dictionaries for their IDs.
        self.raw_sentences = []
        self.raw_role_strings = []
        self.sentence_id_mapping = {}
        self.role_id_mapping = {}

        # Try loading the CSV file using UTF-8 encoding. If it fails due to a UnicodeDecodeError, use ISO-8859-1 encoding.
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="ISO-8859-1")

        # Replace certain characters in the DataFrame with other characters.
        df.replace("~", ",", inplace=True, regex=True)
        df.replace("``", '"', inplace=True, regex=True)
        df.replace("''", '"', inplace=True, regex=True)

        # Sample a fraction of the DataFrame based on the data_ratio argument.
        df = df.sample(frac=data_ratio, random_state=1)

        # Iterate over each row in the DataFrame, displaying progress with a tqdm progress bar.
        for row in tqdm(
            df.itertuples(), total=df.shape[0], desc=f"Loading {file_path}"
        ):
            # Convert the source file name to a unique index.
            source_file_index = self._source_file_to_index(row.source_file)
            # Create a unique ID for the sentence consisting of the source file index and the sentence ID.
            unique_sentence_id = (source_file_index, row.sentence_id)

            # Parse the frame roles in the row.
            frame_roles = self._parse_frame_roles(row.frame_roles, unique_sentence_id)
            # Retrieve the verbatlas mapping for the frame name in the row.
            mapping = self.verb_atlas_frames.get_verbatlas_mapping(row.frame_name)

            # Construct a Frame object using the frame name, lemma name, frame roles, and verbatlas mapping from the row.
            frame = Frame(
                row.frame_name,
                row.lemma_name,
                frame_roles,
                mapping["va_frame_id"],
                mapping["va_frame_name"],
            )

            # If the unique sentence ID is already in the sentences dictionary, append the frame to the frames list of the sentence.
            # Otherwise, create a new Sentence object and add it to the sentences dictionary.
            if unique_sentence_id in self.sentences:
                self.sentences[unique_sentence_id].frames.append(frame)
            else:
                self.sentences[unique_sentence_id] = Sentence(
                    unique_sentence_id, row.sentence_string, [frame]
                )

            # Add each role in frame_roles to the roles dictionary, using the role's ID as the key.
            for role in frame_roles:
                self.roles[role.role_id] = role

            # Add the sentence to the appropriate split (train, dev, or test) based on the split argument.
            getattr(self, split)[unique_sentence_id] = self.sentences[
                unique_sentence_id
            ]

            # Add the sentence string to the raw_sentences list and map its index to its unique ID.
            self.raw_sentences.append(row.sentence_string)
            self.sentence_id_mapping[len(self.raw_sentences) - 1] = unique_sentence_id

            # For each role in frame_roles, add its string to the raw_role_strings list and map its index to its role ID.
            for role in frame_roles:
                self.raw_role_strings.append(role.string)
                self.role_id_mapping[len(self.raw_role_strings) - 1] = role.role_id

    def _process_role_strings(self, role_string: str) -> str:
        # preprocess the role string by removing double quotation marks at the start
        # if role_string.startswith('"'):
        #     role_string = role_string[1:]
        return role_string

    def get_sentence(self, sentence_id: Tuple[int, int]) -> Optional[Sentence]:
        """
        Retrieve a Sentence object from the sentences dictionary by its ID.

        Args:
            sentence_id (Tuple[int, int]): The ID of the sentence, composed of the source file index and the sentence's original ID.

        Returns:
            Optional[Sentence]: The Sentence object corresponding to the given ID. If no sentence with the given ID is found, returns None.
        """
        # Return the Sentence object from the sentences dictionary with the given sentence_id as the key.
        # If there is no such key in the dictionary, the get() method will return None.
        return self.sentences.get(sentence_id)

    def get_sentences_with_va_frame(self, va_frame: str) -> List[Tuple[int, int]]:
        """
        Retrieve a list of sentence IDs for sentences that contain a specific VerbAtlas frame.

        Args:
            va_frame (str): The name of the VerbAtlas frame to search for.

        Returns:
            List[Tuple[int, int]]: A list of sentence IDs (tuples) for sentences that contain the given VerbAtlas frame.
        """
        # Initialize an empty list to hold the IDs of sentences that contain the given VerbAtlas frame.
        sentence_ids = []

        # Iterate over each entry in the sentences dictionary. The items() method returns key-value pairs, so sentence_id is the key
        # and sentence is the value for each dictionary entry.
        for sentence_id, sentence in self.sentences.items():
            # Further iterate over each Frame object in the sentence's frames list.
            for frame in sentence.frames:
                # If the current Frame object's VerbAtlas frame name matches the given va_frame name, add the sentence's ID to the
                # sentence_ids list.
                if frame.va_frame_name == va_frame:
                    sentence_ids.append(sentence_id)

                    # Once a matching frame is found in the sentence, break out of the inner loop to avoid adding the same sentence
                    # ID multiple times in case it contains more than one frame with the same name.
                    break

        # Return the list of sentence IDs. If no sentences with the given VerbAtlas frame were found, this will be an empty list.
        return sentence_ids

    def save_corpus(self, file_path: str) -> None:
        """
        Save the corpus object to a file. This method saves both non-SpaCy data (basic data structure)
        and SpaCy data (sentence and role documents).

        Args:
            file_path (str): The path of the file to which the corpus will be saved.
        """
        # Open the file in binary write mode. This is required for serialization using pickle.
        with open(file_path, "wb") as f:
            # Serialize the corpus data using pickle.dump(), which writes serialized data to a file.
            # The serialized data is a dictionary containing the sentences dictionary, the train, dev, test sets,
            # and the role_id_map.
            pickle.dump(
                {
                    "sentences": self.sentences,
                    "train": self.train,
                    "dev": self.dev,
                    "test": self.test,
                    "role_id_map": self.role_id_map,
                },
                f,
            )
        # Print a confirmation message that the non-SpaCy data was saved.
        print(f"Corpus (non-SpaCy data) saved to {file_path}")

        # Open a separate file in binary write mode for storing the SpaCy data.
        with open(str(file_path) + ".spacy", "wb") as f:
            # Iterate over all sentences in the sentences dictionary.
            for sentence in self.sentences.values():
                # Convert the SpaCy Doc object of the sentence to bytes for serialization.
                serialized_sentence = sentence.doc.to_bytes()
                # Write the size of the serialized sentence to the file before writing the sentence itself.
                # This will help when deserializing, as we will know how many bytes to read for each sentence.
                f.write(len(serialized_sentence).to_bytes(4, "big"))
                # Write the serialized sentence to the file.
                f.write(serialized_sentence)

                # Now process the roles within each frame of the sentence.
                for frame in sentence.frames:
                    for role in frame.roles:
                        # Use a try-except block to handle potential errors during serialization.
                        try:
                            # Convert the SpaCy Doc object of the role to bytes for serialization.
                            serialized_role = role.doc.to_bytes()
                        except Exception as e:
                            # If an error occurs during serialization, reprocess the role with SpaCy and try serializing again.
                            role.doc = self.nlp(role.string)
                            serialized_role = role.doc.to_bytes()

                        # After the serialization is successful, write the size of the serialized role and the serialized role itself to the file.
                        f.write(len(serialized_role).to_bytes(4, "big"))
                        f.write(serialized_role)
            # Print a confirmation message that the SpaCy data was saved.
            print(f"Corpus (SpaCy data) saved to {file_path}.spacy")

    @classmethod
    def load_corpus(cls, file_path: str) -> "Corpus":
        """
        Load a corpus object from a file. This method loads both non-SpaCy data (basic data structure)
        and SpaCy data (sentence and role documents).

        Args:
            file_path (str): The path of the file from which the corpus will be loaded.

        Returns:
            Corpus: The loaded Corpus object.
        """
        # Open the file containing the non-SpaCy data in binary read mode.
        # This is necessary because the data was written in binary mode.
        with open(file_path, "rb") as f:
            # Deserialize the data using pickle.load(). This will return a dictionary.
            data = pickle.load(f)

        # Create a new instance of the Corpus class without calling its __init__ method.
        # This is necessary because we are loading data for an existing instance, not creating a new one.
        corpus = cls.__new__(cls)

        # Assign the loaded data to the corresponding attributes of the Corpus object.
        corpus.sentences = data["sentences"]
        corpus.train = data["train"]
        corpus.dev = data["dev"]
        corpus.test = data["test"]
        corpus.role_id_map = data["role_id_map"]

        # Print a confirmation message that the non-SpaCy data was loaded.
        print(f"Corpus (non-SpaCy data) loaded from {file_path}")

        # Now load the SpaCy data.
        with open(str(file_path) + ".spacy", "rb") as f:
            # Load the SpaCy model. This is necessary to create Doc objects from serialized data.
            nlp = spacy.load("en_core_web_sm")

            # For each sentence in the sentences dictionary, load the corresponding SpaCy Doc object.
            for sentence in corpus.sentences.values():
                # Read the size of the serialized sentence from the file. This was written before the sentence itself.
                length = int.from_bytes(f.read(4), "big")
                # Read the serialized sentence from the file and create a Doc object from it.
                sentence.doc = Doc(nlp.vocab).from_bytes(f.read(length))

                # Now load the SpaCy Doc objects for the roles within each frame of the sentence.
                for frame in sentence.frames:
                    for role in frame.roles:
                        # Read the size of the serialized role from the file.
                        length = int.from_bytes(f.read(4), "big")
                        # Read the serialized role from the file and create a Doc object from it.
                        role.doc = Doc(nlp.vocab).from_bytes(f.read(length))

        # Print a confirmation message that the SpaCy data was loaded.
        print(f"Corpus (SpaCy data) loaded from {file_path}.spacy")

        # Return the loaded Corpus object.
        return corpus

    def get_role_by_id(self, role_id: int) -> Optional[Role]:
        """
        Retrieve a Role object from the role_id_map dictionary using its unique identifier.

        Args:
            role_id (int): The unique identifier of the Role object. This was assigned during the construction of the Corpus.

        Returns:
            Optional[Role]: The Role object with the specified role_id. If no such Role exists, None is returned.
        """
        # Access the role_id_map dictionary attribute of the Corpus object and use the get() method
        # to retrieve the Role object with the given role_id.
        # The get() method is used instead of the [] operator because it will not raise an error
        # if the role_id does not exist in the dictionary. Instead, it will return None.
        return self.role_id_map.get(role_id)

    def visualize_sentence_table(self, sentence_id: Tuple[int, int]) -> None:
        """
        Visualizes the roles and associated strings within the frames of a sentence in a tabular format.

        Args:
            sentence_id (Tuple[int, int]): The unique identifier for a sentence in the Corpus.

        Returns:
            None
        """
        # Retrieve the Sentence object with the provided sentence_id using the get_sentence() method of the Corpus.
        sentence = self.get_sentence(sentence_id)
        # Print the string representation of the sentence to provide context for the tabular representation.
        print(f"Sentence: {sentence.sentence_string}\n")

        # Initialize an empty string to store the HTML for each frame's table.
        table_html = ""

        # Iterate over each Frame in the sentence.
        for frame in sentence.frames:
            # The headers for the table will be the sentence_id and the frame name.
            table_headers = [f"ID: {sentence.sentence_id}", frame.frame_name]

            # Construct the data for the table. For each Role in the Frame, if the va_role attribute is not "unknown",
            # the data row will consist of the role and va_role, and the string attribute of the Role.
            # If the va_role is "unknown", the data row will consist of just the role and the string.
            table_data = [
                [f"{role.role} ({role.va_role})", role.string]
                if role.va_role != "unknown"
                else [role.role, role.string]
                for role in frame.roles
            ]

            # Use the tabulate function from the tabulate library to generate an HTML table from the data.
            # The table headers and data are provided as arguments.
            html_table = tabulate(table_data, headers=table_headers, tablefmt="html")

            # Append the HTML for this table to the table_html string, with a line break between each table.
            table_html += f"{html_table}<br>"

        # Display the HTML using the display() function from the IPython.core.display library and the HTML() class
        # from the same library to properly format the string as HTML.
        display(HTML(table_html))

    def visualize_sentence_displacy(self, sentence_id: Tuple[int, int]) -> None:
        """
        Visualizes the roles within the frames of a sentence using the SpaCy displacy visualizer.

        Args:
            sentence_id (Tuple[int, int]): The unique identifier for a sentence in the Corpus.

        Returns:
            None
        """
        # Retrieve the Sentence object with the provided sentence_id using the get_sentence() method of the Corpus.
        sentence = self.get_sentence(sentence_id)

        # Create an instance of the Spacy NLP model.
        nlp = spacy.load("en_core_web_sm")

        # Generate a Doc object from the sentence_string attribute of the Sentence.
        doc = nlp(sentence.sentence_string)

        # Create an HTML header to describe the output. This includes the number of frames in the sentence and the sentence ID.
        table_header = f"<h3>{len(sentence.frames)} frames for sentence ID: {sentence.sentence_id}</h3>"
        # Display this header using the IPython HTML display function.
        display(HTML(table_header))

        # Iterate over each Frame in the sentence.
        for frame in sentence.frames:
            # Initialize an empty list to store the entities (roles) for the displacy visualizer.
            entities = []

            # Iterate over each Role in the frame.
            for role in frame.roles:
                # Find the start and end indices of the role string within the sentence string.
                role_start = sentence.sentence_string.find(role.string)
                role_end = role_start + len(role.string)

                # Combine the propbank_roles and va_roles with a '/' to form the label for the displacy visualizer,
                # unless the va_role is 'unknown', in which case just use the propbank role.
                label = (
                    f"{role.role} ({role.va_role})"
                    if role.va_role != "unknown"
                    else role.role
                )

                # Append the entity data to the entities list. This data includes the start and end indices of the entity
                # in the sentence, and the label for the entity.
                entities.append(
                    {
                        "start": role_start,
                        "end": role_end,
                        "label": label,
                    }
                )

            # Prepare the data to be visualized by the displacy render method. This includes the text of the sentence,
            # the entities within the sentence, and a title for the visualization that includes the sentence_id and frame name.
            render_data = [
                {
                    "text": doc.text,
                    "ents": entities,
                    "title": f"ID: {sentence.sentence_id} {frame.frame_name}",
                }
            ]

            # Define a color mapping for different roles. This will be used to color the entities in the displacy visualization.
            color_mapping = {
                "ARG0": "#FCA311",
                "ARG1": "#2EC4B6",
                "ARG2": "#E63946",
                "ARG3": "#DD6E42",
                "ARG4": "#4EA8DE",
                "FEE": "#57A773",
                "V": "#57A773",
            }

            # Initialize an empty dictionary to store the final colors for each label.
            colors = {}
            # Iterate over each Role in the Frame again to assign a color to each label.
            for role in frame.roles:
                # Determine the label for the Role as before.
                label = (
                    f"{role.role} ({role.va_role})"
                    if role.va_role != "unknown"
                    else role.role
                )
                # Get the color for this role from the color_mapping dictionary. If the role is not in the dictionary,
                # use a default color.
                color = color_mapping.get(role.role, "#EA8189")
                # Assign this color to the label in the colors dictionary.
                colors[label] = color

            # Define the options for the displacy visualizer. These options include the compact mode, offset and distance
            # between entities, manual mode, fine-grained mode, and the color for each entity.
            displacy_options = {
                "compact": True,
                "offset_x": 100,
                "distance": 100,
                "manual": True,
                "fine_grained": True,
                "colors": colors,
            }

            # Use the displacy.render method from the SpaCy library to generate the HTML for the visualization.
            # The render data, style, options, and other parameters are provided as arguments.
            html = displacy.render(
                render_data,
                style="ent",
                manual=True,
                options=displacy_options,
                page=True,
                jupyter=False,
                minify=True,
            )

            # Display the HTML using the IPython HTML display function.
            display(HTML(html))

    def get_modifiers_df(self, corpus_part: str) -> pd.DataFrame:
        """
        Returns a pandas DataFrame of all roles that are not ARG0, ARG1, ARG2, ARG3, ARG4, ARG5, V
        in the specified part of the corpus.
        """
        # Make sure the specified corpus part is valid
        assert corpus_part in {
            "full",
            "train",
            "dev",
            "test",
        }, "Invalid corpus part. Please choose 'train', 'dev', or 'test'."

        # Select the specified part of the corpus
        if corpus_part == "train":
            corpus_data = self.train
        elif corpus_part == "dev":
            corpus_data = self.dev
        elif corpus_part == "test":
            corpus_data = self.test
        elif corpus_part == "full":
            corpus_data = self.sentences

        # Specify the roles you want to ignore
        core_roles = {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5", "ARGA", "V"}

        # Initialize a list to store the rows of the DataFrame
        data = []

        # Iterate over all sentences in the selected part of the corpus
        for sentence_id, sentence in corpus_data.items():
            # For each sentence, iterate over all frames
            for frame in sentence.frames:
                # For each frame, check the role of each role object
                for role in frame.roles:
                    # If the role is not in the specified set, save it along with its attributes and the sentence_id
                    if role.role not in core_roles:
                        data.append(
                            {
                                "sentence_id": sentence_id,
                                "role": role.role,
                                "indices": role.indices,
                                "string": role.string,
                            }
                        )

        # Convert the list to a DataFrame
        df = pd.DataFrame(data)

        return df
