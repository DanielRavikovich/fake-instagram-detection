"""
src/preprocessing.py
--------------------
Data loading, merging, cleaning, and feature engineering.
Extracted from the project notebook - same logic, same variable names.
"""

import json
import pandas as pd
import numpy as np


# Columns shared between InstaFake and Kaggle after alignment
cols = ['profile pic', 'nums/length username', 'description length',
        'private', '#posts', '#followers', '#follows', 'fake']

# Column name mapping: InstaFake JSON -> unified format
rename_map = {
    'userFollowerCount':   '#followers',
    'userFollowingCount':  '#follows',
    'userBiographyLength': 'description length',
    'userMediaCount':      '#posts',
    'userHasProfilPic':    'profile pic',
    'userIsPrivate':       'private',
    'isFake':              'fake'
}


def load_instafake(fake_path='fakeAccountData.json', real_path='realAccountData.json'):
    """Load and harmonise the InstaFake JSON files."""
    with open(fake_path, 'r') as f:
        fake_json = json.load(f)
    with open(real_path, 'r') as f:
        real_json = json.load(f)

    df_fake_insta = pd.DataFrame(fake_json)
    df_real_insta = pd.DataFrame(real_json)

    # calculating nums/length username
    df_fake_insta['nums/length username'] = (
        df_fake_insta['usernameDigitCount'] / df_fake_insta['usernameLength']
    ).fillna(0)
    df_real_insta['nums/length username'] = (
        df_real_insta['usernameDigitCount'] / df_real_insta['usernameLength']
    ).fillna(0)

    # column name change to a uniform format
    df_fake_insta = df_fake_insta.rename(columns=rename_map)
    df_real_insta = df_real_insta.rename(columns=rename_map)

    # adding 'fake' column to the JSON files like in the CSV files
    df_fake_insta['fake'] = 1
    df_real_insta['fake'] = 0

    # merging the JSON files to one file
    df_instafake = pd.concat(
        [df_fake_insta[cols], df_real_insta[cols]], ignore_index=True
    )
    return df_instafake


def load_kaggle(train_path='train.csv', test_path='test.csv'):
    """Load and merge the Kaggle train/test CSV files."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # merging train and test csv (will split later)
    df_kaggle = pd.concat([df_train, df_test], ignore_index=True)

    # saving mutual columns with the JSON
    df_kaggle = df_kaggle[cols]
    return df_kaggle


def build_dataset(fake_path='fakeAccountData.json', real_path='realAccountData.json',
                  train_path='train.csv', test_path='test.csv'):
    """
    Load, merge, and clean both data sources.
    Returns a cleaned DataFrame with duplicate rows removed.
    """
    df_instafake = load_instafake(fake_path, real_path)
    df_kaggle = load_kaggle(train_path, test_path)

    # final merge
    df = pd.concat([df_instafake, df_kaggle], ignore_index=True)
    print("Final dataset shape:", df.shape)

    # Remove duplicate rows and reset index
    df = df.drop_duplicates().reset_index(drop=True)
    print("Shape after removing duplicates:", df.shape)
    print("\nClass distribution after deduplication:")
    print(df['fake'].value_counts())

    return df


def add_ratio_features(df):
    """
    Add the three ratio features from the paper, with Laplace smoothing (+1)
    on the denominator to avoid division by zero.
    """
    df = df.copy()
    df['following/followers ratio'] = df['#follows'] / (df['#followers'] + 1)
    df['following/posts ratio'] = df['#follows'] / (df['#posts'] + 1)
    df['followers/posts ratio'] = df['#followers'] / (df['#posts'] + 1)
    return df
