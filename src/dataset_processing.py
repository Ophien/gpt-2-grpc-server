import re
import sys

def process_dataset(path):
    dataset = open(path, "r+")
    dataset_string = dataset.read()

    # remove unwanted referencing for papers and wikipedia text
    dataset_string = re.sub(r"\\|\'|\*|\_|\{.*\}|\#|\+|\-*|\$|\"|\[.*\](:([0-9]*-*[0-9]*))*|\(.*\)", '', dataset_string)
    dataset_string = re.sub(r"[ \f\t\v]$", '', dataset_string)
    dataset_string

    dataset.seek(0)
    dataset.write(dataset_string)
    dataset.truncate()
    dataset.close()