import os  # nopep8
import sys  # nopep8
from pathlib import Path  # nopep8

# Add the 'src' directory to sys.path
current_working_directory = os.getcwd()  # Get the current working directory
src_path = (
    Path(current_working_directory) / "src"
)  # Construct the path to the 'src' directory
sys.path.append(str(src_path))  # Add 'src' directory to sys.path

from corpus import *

# Define the data directory path
data_dir = Path(current_working_directory) / "data"
corpus_data_dir = data_dir / "ontonotes_data"
verbatlas_data_dir = data_dir / "verbatlas_data"

# Load corpus data
train_corpus_file_path = corpus_data_dir / "ontonotes_train.csv"
dev_corpus_file_path = corpus_data_dir / "ontonotes_dev.csv"
test_corpus_file_path = corpus_data_dir / "ontonotes_test.csv"

pb2va_file_path = verbatlas_data_dir / "pb2va.tsv"
VA_frame_info_file_path = verbatlas_data_dir / "VA_frame_info.tsv"
verb_atlas_frames = VerbAtlasFrames(pb2va_file_path, VA_frame_info_file_path)

corpus = Corpus(
    train_corpus_file_path,
    dev_corpus_file_path,
    test_corpus_file_path,
    verb_atlas_frames,
    data_ratio=0.01,
)

corpus.save_corpus("test_corpus_ontonotes.pkl")

# corpus = Corpus.load_corpus("corpus.pkl")
