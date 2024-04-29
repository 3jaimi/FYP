import os
import tensorflow as tf
import sys

def count_records_in_tfrecords(tfrecords_dir: str) -> int:
    """
    Counts the total number of records in all TFRecords files in the specified directory.

    :param tfrecords_dir: Path to the directory containing TFRecords files.
    :return: Total number of records in all TFRecords files.
    """
    total_records = 0

    try:
        # Get a list of all files in the directory
        file_list = os.listdir(tfrecords_dir)

        # Filter only the TFRecords files
        tfrecord_files = [file for file in file_list if file.lower().endswith(".tfrecords")]

        # Iterate through each TFRecords file and count records
        for tfrecord_file in tfrecord_files:
            file_path = os.path.join(tfrecords_dir, tfrecord_file)
            record_iterator = tf.data.TFRecordDataset(file_path).as_numpy_iterator()
            num_records = sum(1 for _ in record_iterator)
            total_records += num_records

        return total_records
    except FileNotFoundError:
        print(f"Error: Directory '{tfrecords_dir}' not found.")
        return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_tfrecords_records.py <path_to_tfrecords_directory>")
        sys.exit(1)

    tfrecords_directory = sys.argv[1]
    num_records = count_records_in_tfrecords(tfrecords_directory)
    print(f"Total records in TFRecords files in '{tfrecords_directory}': {num_records}")
