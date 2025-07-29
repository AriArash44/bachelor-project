import pandas as pd

def process_ip_csv_to6(input_csv_path: str) -> str:
    df = pd.read_csv(input_csv_path)
    def split_into_6(ip: str) -> list[int]:
        if pd.isna(ip):
            return [-1] * 6
        if ":" in ip:
            parts = ip.split(":")
            parts = parts + [""] * (6 - len(parts)) if len(parts) < 6 else parts[:6]
            nums = []
            for p in parts:
                if p == "":
                    nums.append(-1)
                else:
                    try:
                        nums.append(int(p, 16))
                    except ValueError:
                        nums.append(-1)
            return nums
        parts = ip.split(".")
        nums = []
        for p in parts[:4]:
            try:
                nums.append(int(p))
            except ValueError:
                nums.append(-1)
        nums += [-1] * (6 - len(nums))
        return nums
    for col in ["src_ip", "dst_ip"]:
        if col not in df.columns:
            continue
        exploded = df[col].apply(split_into_6)
        for idx in range(6):
            df[f"{col}.part{idx+1}"] = exploded.map(lambda lst: lst[idx])
        df.drop(columns=[col], inplace=True)
    output_path = "./ip_processed.csv"
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    process_ip_csv_to6("../../0-testTrainSplitter/train_split.csv")