import pandas as pd
import numpy as np
import random
import argparse

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
    linux_path:   str,
    network_path: str,
    num_windows:  int,
    output_csv:   str
):
    df_dev = pd.read_csv(devices_path)
    df_linux = pd.read_csv(linux_path)
    df_network = pd.read_csv(network_path)

    dev_wins = slice_windows(df_dev)
    linux_wins = slice_windows(df_linux)
    network_wins = slice_windows(df_network)

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
        idx = random.randrange(max_windows)
        sub_dev = dev_wins[idx].copy()
        sub_lin = linux_wins[idx].copy()
        sub_net = network_wins[idx].copy()
        flags = {"dev":"normal", "lin":"normal", "net":"normal"}
        for ds in pick_map[lbl]:
            if ds == "dev":
                flags["dev"] = lbl
            elif ds == "lin":
                flags["lin"] = lbl
            else:
                flags["net"] = lbl
        sub_dev.columns = [f"{c}_dev" for c in sub_dev.columns]
        sub_lin.columns = [f"{c}_lin" for c in sub_lin.columns]
        sub_net.columns = [f"{c}_net" for c in sub_net.columns]
        combo = pd.concat([sub_dev, sub_lin, sub_net], axis=1)
        combo["device_label"] = flags["dev"]
        combo["linux_label"] = flags["lin"]
        combo["network_label"] = flags["net"]
        combo["final_label"] = lbl
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
