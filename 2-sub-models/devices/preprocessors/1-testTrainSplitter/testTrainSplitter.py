import pandas as pd
import numpy as np

def extract_non_overlapping_blocks(df, block_size=20):
    total_rows = len(df)
    max_start = total_rows - block_size
    step = block_size
    possible_starts = list(range(0, max_start + 1, step))
    np.random.shuffle(possible_starts)
    blocks = []
    for start in possible_starts:
        block = df.iloc[start:start + block_size]
        blocks.append(block)
    return blocks

def split_blocks(blocks, train_ratio=0.8):
    np.random.shuffle(blocks)
    split_idx = int(len(blocks) * train_ratio)
    train_blocks = blocks[:split_idx]
    test_blocks = blocks[split_idx:]
    train_df = pd.concat(train_blocks).reset_index(drop=True)
    test_df = pd.concat(test_blocks).reset_index(drop=True)
    return train_df, test_df

source_file = "../0-logAggregator/aggregated.csv"
df = pd.read_csv(source_file)
blocks = extract_non_overlapping_blocks(df, block_size=20)
train_df, test_df = split_blocks(blocks, train_ratio=0.8)
train_df.to_csv("train_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)
print("Created train_split.csv and test_split.csv.")