import pandas as pd
import numpy as np
import random
import argparse
from collections import Counter, defaultdict

def safe_choice(bins_dict, target_label):
    lst = bins_dict.get(target_label, [])
    if not lst:
        lst = bins_dict.get("normal", [])
    return random.choice(lst)

def group_windows_by_majority_label(wins, label_col):
    bins = defaultdict(list)
    for idx, win in enumerate(wins):
        most_common_label = Counter(win[label_col]).most_common(1)[0][0]
        bins[most_common_label].append(idx)
    return bins

def calculFinalLabel(devicesLabel, linuxLabel, networkLabel):
    if(devicesLabel == linuxLabel == networkLabel):
        return devicesLabel
    if devicesLabel == networkLabel and devicesLabel != "normal":
        return devicesLabel
    elif linuxLabel == networkLabel and linuxLabel != "normal":
        return linuxLabel
    elif devicesLabel == linuxLabel and devicesLabel != "normal":
        return devicesLabel
    elif networkLabel != "normal":
        return networkLabel
    elif devicesLabel != "normal":
        return devicesLabel
    elif linuxLabel != "normal":
        return linuxLabel
    
window_size = 20
def slice_windows(df):
    Nwin = len(df) // window_size
    return [df.iloc[i*window_size:(i+1)*window_size].reset_index(drop=True) for i in range(Nwin)]

def generate_combined_dataset(
    devices_path: str,
    linux_path: str,
    network_path: str,
    num_windows: int,
    output_csv: str
):
    df_dev = pd.read_csv(devices_path)
    df_linux = pd.read_csv(linux_path)
    df_network = pd.read_csv(network_path)

    dev_wins = slice_windows(df_dev)
    linux_wins = slice_windows(df_linux)
    network_wins = slice_windows(df_network)

    dev_bins = group_windows_by_majority_label(dev_wins, "type")
    linux_bins = group_windows_by_majority_label(linux_wins, "type")
    net_bins = group_windows_by_majority_label(network_wins, "type")

    max_windows = min(len(dev_wins), len(linux_wins), len(network_wins))
    if num_windows > max_windows:
        raise ValueError(f"maximum windows to create: {max_windows}")

    LABELS = ["normal", "xss", "mitm", "ransomware", "dos", "ddos", "scanning", "password", "injection", "backdoor"]
    reps = num_windows // len(LABELS)
    rem = num_windows %  len(LABELS)
    labels = LABELS * reps + random.sample(LABELS, rem)
    random.shuffle(labels)

    pick_map = {
        "normal": ["dev","lin","net"],
        "xss": ["dev","lin","net"],
        "scanning": ["dev","lin","net"],
        "password": ["dev","lin","net"],
        "injection": ["dev","lin","net"],
        "ddos": ["dev","lin","net"],
        "mitm": ["lin","net"],
        "dos": ["lin","net"],
        "ransomware": ["dev","net"],
        "backdoor": ["dev","net"],
    }

    final_dfs = []
    for lbl in labels:
        targets = {
            "dev": lbl if "dev" in pick_map[lbl] else "normal",
            "lin": lbl if "lin" in pick_map[lbl] else "normal",
            "net": lbl if "net" in pick_map[lbl] else "normal",
        }

        idx_dev = safe_choice(dev_bins, targets["dev"])
        idx_lin = safe_choice(linux_bins, targets["lin"])
        idx_net = safe_choice(net_bins, targets["net"])

        sub_dev = dev_wins[idx_dev].copy()
        sub_lin = linux_wins[idx_lin].copy()
        sub_net = network_wins[idx_net].copy()

        temp = pd.DataFrame({
            "device_label": sub_dev["type"],
            "linux_label": sub_lin["type"],
            "network_label": sub_net["type"],
        })
        temp["type"] = temp.apply(
            lambda row: calculFinalLabel(
                row["device_label"],
                row["linux_label"],
                row["network_label"]
            ),
            axis=1
        )

        sub_dev = sub_dev.drop(columns=["type"])
        sub_lin = sub_lin.drop(columns=["type"])
        sub_net = sub_net.drop(columns=["type"])

        combo = pd.concat([sub_dev, sub_lin, sub_net], axis=1)
        combo["type"] = temp["type"]

        final_dfs.append(combo)

    result_df = pd.concat(final_dfs, ignore_index=True)
    result_df.to_csv(output_csv, index=False)
    print(f"result saved in: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="merging all three parts of dataset together to mimic activities dataset")
    parser.add_argument("devices_path", help="devices dataset path")
    parser.add_argument("linux_path", help="linux dataset path")
    parser.add_argument("network_path", help="network dataset path")
    parser.add_argument("num_windows", type=int, help="number of windows in output csv")
    parser.add_argument("output_csv", help="output csv")
    args = parser.parse_args()
    generate_combined_dataset(
        devices_path=args.devices_path,
        linux_path=args.linux_path,
        network_path=args.network_path,
        num_windows=args.num_windows,
        output_csv=args.output_csv
    )
