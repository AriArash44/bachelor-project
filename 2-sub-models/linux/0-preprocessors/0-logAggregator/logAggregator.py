import pandas as pd
import warnings

csv_files = {
    "disk": "../../../../0-datasets/linux/Train_test_linux_disk.csv",
    "memory": "../../../../0-datasets/linux/Train_test_linux_memory.csv",
    "process": "../../../../0-datasets/linux/Train_Test_Linux_process.csv",
}

label_columns = [
    'disk.attack',
    'memory.label',
    'process.label',
]

type_columns = [
    'disk.type',
    'memory.type',
    'process.type',
]

pid_columns = [
    'disk.PID',
    'memory.PID',
    'process.PID',
]

cmd_columns = [
    'disk.CMD',
    'memory.CMD',
    'process.CMD',
]

def extract_feature(row, columns):
    feature_values = []
    feature_dataset = []
    for col in columns:
        if pd.notnull(row[col]):
            feature_values.append(row[col])
            feature_dataset.append(col)
    return feature_values, feature_dataset

def get_first_feature(row, columns, default_value=None):
    feature_values, feature_dataset = extract_feature(row, columns)
    if feature_values and pd.notnull(feature_values[0]):
        return feature_values[0], feature_dataset[0]
    return default_value, default_value

def integration_checker(pre, new):
    if (
        get_first_feature(new, pid_columns)[0] in extract_feature(pre, pid_columns)[0] and
        get_first_feature(new, cmd_columns)[0] in extract_feature(pre, cmd_columns)[0] and
        get_first_feature(new, label_columns)[0] in extract_feature(pre, label_columns)[0] and
        get_first_feature(new, type_columns)[0] in extract_feature(pre, type_columns)[0]
    ):
        return True
    return False

dataframes = []

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
for prefix, file in csv_files.items():
    df = pd.read_csv(file)
    df = df.add_prefix(f"{prefix}.")
    if 'disk.WRDSK' in df.columns:
        df['disk.WRDSK'] = pd.to_numeric(df['disk.WRDSK'], errors='coerce')
    df = df.drop_duplicates()
    dataframes.append(df)

combined_df = pd.concat(dataframes, axis=0, ignore_index=True)

combined_df["sort_pid"] = combined_df.apply(
    lambda row: get_first_feature(row, pid_columns)[0], axis=1
)

combined_df["sort_cmd"] = combined_df.apply(
    lambda row: get_first_feature(row, cmd_columns)[0], axis=1
)

combined_df["sort_label"] = combined_df.apply(
    lambda row: get_first_feature(row, label_columns)[0], axis=1
)

combined_df["sort_type"] = combined_df.apply(
    lambda row: get_first_feature(row, type_columns)[0], axis=1
)

sorted_df = combined_df.sort_values(by=["sort_cmd", "sort_pid", "sort_label", "sort_type"])
sorted_df.drop(columns=["sort_pid", "sort_cmd", "sort_label", "sort_type"], inplace=True)
sorted_df = sorted_df.reset_index(drop=True)

sorted_df["processed"] = combined_df.apply(
    lambda row: "false", axis=1
)

activity_logs = []
new_activity_flag = True

for idx, row in sorted_df.iterrows():
    if new_activity_flag:
        if row["processed"] == "true":
            continue
        row["processed"] = "true"
        activity_log = row.to_dict()
        activity_logs.append(activity_log)
        new_activity_flag = False
    else:
        previous_log = activity_logs[-1]
        baseIndex = idx
        currentRow = sorted_df.iloc[baseIndex]
        while integration_checker(previous_log, currentRow):
            if get_first_feature(currentRow, label_columns)[1] not in extract_feature(previous_log, label_columns)[1]:
                for col in sorted_df.columns:
                    if col not in previous_log or pd.isnull(previous_log[col]):
                        previous_log[col] = currentRow[col]
                sorted_df.at[baseIndex, "processed"] = "true"
            baseIndex += 1 
            currentRow = sorted_df.iloc[baseIndex]
        previous_log["PID"] = get_first_feature(previous_log, pid_columns)[0]
        previous_log["CMD"] = get_first_feature(previous_log, cmd_columns)[0]
        previous_log["label"] = get_first_feature(previous_log, label_columns)[0]
        previous_log["type"] = get_first_feature(previous_log, type_columns)[0]
        new_activity_flag = True

bucketed_df = pd.DataFrame(activity_logs)
bucketed_df.drop(columns=pid_columns, inplace=True)
bucketed_df.drop(columns=cmd_columns, inplace=True)
bucketed_df.drop(columns=label_columns, inplace=True)
bucketed_df.drop(columns=type_columns, inplace=True)
bucketed_df.drop(columns=["processed"], inplace=True)

bucketed_df.to_csv("aggregated.csv", index=False)

print("Aggregating dataset tasks completed!")