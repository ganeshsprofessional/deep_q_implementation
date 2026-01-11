import json
import csv
import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# -----------------------------
# Hard-coded CSV / DataFrame column order
# -----------------------------
CSV_COLUMNS = [
    "type",
    "rcvTime",
    "sendTime",
    "sender",
    "senderPseudo",
    "messageID",

    "pos_x", "pos_y", "pos_z",
    "pos_noise_x", "pos_noise_y", "pos_noise_z",

    "spd_x", "spd_y", "spd_z",
    "spd_noise_x", "spd_noise_y", "spd_noise_z",

    "acl_x", "acl_y", "acl_z",
    "acl_noise_x", "acl_noise_y", "acl_noise_z",

    "hed_x", "hed_y", "hed_z",
    "hed_noise_x", "hed_noise_y", "hed_noise_z",
]

def get_scalar(data, key):
    """Safely extract scalar value"""
    return data.get(key, "")


def get_vec(data, key, idx):
    """Safely extract vector element"""
    vec = data.get(key)
    if isinstance(vec, list) and idx < len(vec):
        return vec[idx]
    return ""


def json_to_row(data):
    """Map JSON record to fixed row dict"""
    return {
        "type": get_scalar(data, "type"),
        "rcvTime": get_scalar(data, "rcvTime"),
        "sendTime": get_scalar(data, "sendTime"),
        "sender": get_scalar(data, "sender"),
        "senderPseudo": get_scalar(data, "senderPseudo"),
        "messageID": get_scalar(data, "messageID"),

        "pos_x": get_vec(data, "pos", 0),
        "pos_y": get_vec(data, "pos", 1),
        "pos_z": get_vec(data, "pos", 2),

        "pos_noise_x": get_vec(data, "pos_noise", 0),
        "pos_noise_y": get_vec(data, "pos_noise", 1),
        "pos_noise_z": get_vec(data, "pos_noise", 2),

        "spd_x": get_vec(data, "spd", 0),
        "spd_y": get_vec(data, "spd", 1),
        "spd_z": get_vec(data, "spd", 2),

        "spd_noise_x": get_vec(data, "spd_noise", 0),
        "spd_noise_y": get_vec(data, "spd_noise", 1),
        "spd_noise_z": get_vec(data, "spd_noise", 2),

        "acl_x": get_vec(data, "acl", 0),
        "acl_y": get_vec(data, "acl", 1),
        "acl_z": get_vec(data, "acl", 2),

        "acl_noise_x": get_vec(data, "acl_noise", 0),
        "acl_noise_y": get_vec(data, "acl_noise", 1),
        "acl_noise_z": get_vec(data, "acl_noise", 2),

        "hed_x": get_vec(data, "hed", 0),
        "hed_y": get_vec(data, "hed", 1),
        "hed_z": get_vec(data, "hed", 2),

        "hed_noise_x": get_vec(data, "hed_noise", 0),
        "hed_noise_y": get_vec(data, "hed_noise", 1),
        "hed_noise_z": get_vec(data, "hed_noise", 2),
    }


def jsonl_to_dataframe(input_file, output_csv=None):
    """
    Convert JSONL file to pandas DataFrame.
    Optionally write CSV if output_csv is provided.
    """
    rows = []

    with open(input_file, "r") as fin:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            if data["type"] != 2:
                rows.append(json_to_row(data))

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df


def extract_state_vector(df): 
    #calculate S(j,t) and X(j,t)
    spd_cols = [c for c in ["spd_x", "spd_y", "spd_z"] if c in df.columns]
    df["speed"] = np.sqrt((df[spd_cols] ** 2).sum(axis=1))
    acl_cols = [c for c in ["acl_x", "acl_y", "acl_z"] if c in df.columns]
    df["acceleration"] = np.sqrt((df[acl_cols] ** 2).sum(axis=1))
    
    #calculate mu(i,t) and sig(i,t)
    df['rcvTime_td'] = pd.to_timedelta(df['rcvTime'], unit='s')
    df = df.set_index('rcvTime_td')
    df['avg_speed_1s'] = df['speed'].rolling('1s').mean()
    df['stddev_speed_1s'] = df['speed'].rolling('1s').std()
    
    #calulate B(j,t)
    df = df.sort_values(['sender', 'rcvTime'])
    df['msg_count'] = df.groupby('sender').cumcount() + 1
    df['elapsed_time'] = (
        df['rcvTime'] - df.groupby('sender')['rcvTime'].transform('first')
    )
    df['avg_sender_rate'] = df['msg_count'] / df['elapsed_time']
    df.loc[df['elapsed_time'] == 0, 'avg_sender_rate'] = 0.0
    df = df.drop(columns=['msg_count', 'elapsed_time'])
    df = df.sort_values(['sender', 'rcvTime'])
    
    #calculate del(j,t)
    group = df.groupby("sender")
    
    df["prev_x"] = group["pos_x"].shift(1)
    df["prev_y"] = group["pos_y"].shift(1)
    df["prev_z"] = group["pos_z"].shift(1)
    
    df["prev_speed"] = group["speed"].shift(1)
    df["prev_acc"] = group["acceleration"].shift(1)
    df["prev_time"] = group["rcvTime"].shift(1)
    df["dt"] = df["rcvTime"] - df["prev_time"]
    df["euclidean_dist"] = np.sqrt(
        (df["pos_x"] - df["prev_x"])**2 +
        (df["pos_y"] - df["prev_y"])**2 +
        (df["pos_z"] - df["prev_z"])**2
    )
    df["kinematic_dist"] = (
        ((df["prev_speed"] + df["speed"]) / 2) * df["dt"] +
        0.5 * ((df["prev_acc"] + df["acceleration"]) / 2) * df["dt"]**2
    )
    df["distance_diff"] = df["kinematic_dist"] - df["euclidean_dist"]
    df = df.reset_index(drop=True)
    
    output = df[["sender", "rcvTime", "avg_sender_rate", "speed", "acceleration", "distance_diff", "avg_speed_1s", "stddev_speed_1s"]]
    
    return output


def extract_receiver(filename):
    """
    Extracts the second numeric field from filenames like:
    traceJSON-10545-10543-A0-25200-7.json
    """
    match = re.search(r"traceJSON-(\d+)-(\d+)-", filename)
    if match:
        #veremi-extension specific
        return int(match.group(1))
    else:
        return None

def get_dataframe(files):
    dfs = []
    # i = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        receiver = extract_receiver(filename)
        df = jsonl_to_dataframe(file_path)
        # df = extract_state_vector(df)
    
        # --- Add receiver column AFTER extraction ---
        df["receiver"] = receiver
        dfs.append(df)
        
        # i += 1
        # if i > 5: 
        #     break
    
    
    # Combine all files into one DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

def get_features_logs(files):
    dfs = []
    # i = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        receiver = extract_receiver(filename)
        df = jsonl_to_dataframe(file_path)
        df = extract_state_vector(df)
    
        # --- Add receiver column AFTER extraction ---
        df["receiver"] = receiver
        dfs.append(df)
        
        # i += 1
        # if i > 5: 
        #     break
    
    
    # Combine all files into one DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

