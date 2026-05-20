import json
import argparse
import io
import pandas as pd
import json
import sys
import math
from tqdm import tqdm




def generate_qa_pairs(main_data, dataset_name):
    # Sort the dataframe by UserId, pseudo_session_trajectory_id, and timestamp
    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])

    # List to store the QA pairs
    qa_pairs = []

    # Iterate over each user
    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]

        # Iterate over each unique trajectory for the user based on 'pseudo_session_trajectory_id'
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]
            user_trajectory_data.reset_index(drop=True, inplace=True)

            # Create the question based on the current trajectory (excluding the last entry) and historical data
            question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
            for i, row in user_trajectory_data.iloc[:-1].iterrows():
                if i > 0:
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a "
                        f"{row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")
                else:
                    question_parts = [f"<question>: The following data is a trajectory of user {user}:"]
                    question_parts.append(
                        f"At {row['UTCTimeOffset']}, user {user} visited POI id {row['PoiId']} which is a "
                        f"{row['PoiCategoryName']} and has Category id {row['PoiCategoryId']}.")

            # Create the final question string
            question = " ".join(question_parts)
            value = {'nyc': 4981, 'tky': 7833, 'ca': 9690}[dataset_name]
            question += (f" Given the data, At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, Which POI id will "
                         f"user {user} visit? Note that POI id is an integer in the range from 0 to {value}.")

            # Form the answer based on the last entry of the current trajectory
            answer = (f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will "
                      f"visit POI id {user_trajectory_data.iloc[-1]['PoiId']}.")

            category = (f"<category>: {user_trajectory_data.iloc[-1]['PoiCategoryName']}")

            # Append the question-answer pair to the list
            qa_pairs.append((question, answer, category))
    return qa_pairs

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def preprocess_stage2(dataset_name):
    print(f"Processing dataset: {dataset_name}")
    path = f'datasets/{dataset_name}/preprocessed/'

    # Read the data
    train_data = pd.read_csv(f'{path}train_sample.csv')
    test_data = pd.read_csv(f'{path}test_sample_with_traj.csv')
    val_data = pd.read_csv(f'{path}validate_sample_with_traj.csv')

    # Generate the QA pairs
    qa_pairs_train = generate_qa_pairs(train_data, dataset_name)
    qa_pairs_test = generate_qa_pairs(test_data, dataset_name)
    qa_pairs_val = generate_qa_pairs(val_data, dataset_name)

    # Save the train QA pairs in JSON format
    qa_dict_train = [{"question": q, "answer": a, "category": c} for q, a, c in qa_pairs_train]
    with open(f'{path}train_qa_pairs_kqt.json', 'w') as json_file:
        json.dump(qa_dict_train, json_file)

    # Save the test and val QA pairs in TXT format
    with open(f'{path}test_qa_pairs_kqt.txt', 'w') as txt_file:
        for q, a, c in qa_pairs_test:
            txt_file.write(q + a + c + '\n')

    with open(f'{path}val_qa_pairs_kqt.txt', 'w') as txt_file_2:
        for q, a, c in qa_pairs_val:
            txt_file_2.write(q + a + c + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=['ca', 'nyc', 'tky'],
                        help="Name of the dataset (e.g., ca, nyc, tky)")
    args = parser.parse_args()
    preprocess_stage2(args.dataset_name)

