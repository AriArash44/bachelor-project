import pandas as pd

csv_files = {
    "fridge": "../../../../0-datasets/iot devices/Train_Test_IoT_Fridge.csv",
    "garage_door": "../../../../0-datasets/iot devices/Train_Test_IoT_Garage_Door.csv",
    "gps_tracker": "../../../../0-datasets/iot devices/Train_Test_IoT_GPS_Tracker.csv",
    "modbus": "../../../../0-datasets/iot devices/Train_Test_IoT_Modbus.csv",
    "motion_light": "../../../../0-datasets/iot devices/Train_Test_IoT_Motion_Light.csv",
    "thermostat": "../../../../0-datasets/iot devices/Train_Test_IoT_Thermostat.csv",
    "weather": "../../../../0-datasets/iot devices/Train_Test_IoT_Weather.csv"
}

time_columns = [
    "fridge.datetime", "garage_door.datetime", "gps_tracker.datetime",
    "modbus.datetime", "motion_light.datetime", "thermostat.datetime",
    "weather.datetime"
]

type_columns = [
    'fridge.type', 'garage_door.type', 'gps_tracker.type', 'modbus.type',
    'motion_light.type', 'thermostat.type', 'weather.type'
]

label_columns = [
    'fridge.label', 'garage_door.label', 'gps_tracker.label', 'modbus.label',
    'motion_light.label', 'thermostat.label', 'weather.label'
]

def extract_feature(row, columns):
    vals, keys = [], []
    for c in columns:
        v = row[c]
        if pd.notnull(v):
            vals.append(v)
            keys.append(c)
    return vals, keys

def get_first_feature(row, columns, default=None):
    vals, keys = extract_feature(row, columns)
    if vals:
        return vals[0], keys[0]
    return default, default

def integration_checker(prev, new):
    new_val, new_key = get_first_feature(new, type_columns)
    prev_vals, prev_keys = extract_feature(prev, type_columns)
    return (new_val in prev_vals) and (new_key not in prev_keys)

dfs = []
for prefix, path in csv_files.items():
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(
        df["date"].str.strip() + " " + df["time"].str.strip(),
        format="%d-%b-%y %H:%M:%S", errors="raise"
    )
    df.drop(["date", "time"], axis=1, inplace=True)
    df = df.add_prefix(f"{prefix}.")
    cols = [f"{prefix}.datetime"] + [c for c in df.columns if c != f"{prefix}.datetime"]
    dfs.append(df[cols])

combined = pd.concat(dfs, ignore_index=True)
for col in time_columns:
    combined[col] = pd.to_datetime(combined[col], errors='coerce')

combined["sort_ts"] = combined.apply(
    lambda r: get_first_feature(r, time_columns)[0], axis=1
)
sorted_df = combined.sort_values("sort_ts").reset_index(drop=True)
sorted_df.drop("sort_ts", axis=1, inplace=True)

activity_logs = []
for i, row in sorted_df.iterrows():
    if i == 0:
        v, _ = get_first_feature(row, type_columns)
        log = row.to_dict()
        log["type"] = v
        activity_logs.append(log)
        continue
    prev = activity_logs[-1]
    if integration_checker(prev, row):
        for c in sorted_df.columns:
            if c not in prev or pd.isnull(prev[c]):
                prev[c] = row[c]
    else:
        prev["type"], _ = get_first_feature(prev, type_columns)
        new_type, _ = get_first_feature(row, type_columns)
        if new_type is not None:
            log = row.to_dict()
            log["type"] = new_type
            activity_logs.append(log)

bucketed_df = pd.DataFrame(activity_logs)
bucketed_df.drop(columns=type_columns, inplace=True)
bucketed_df.drop(columns=label_columns, inplace=True)

for col in time_columns:
    bucketed_df[col] = bucketed_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

bucketed_df.to_csv("aggregated.csv", index=False)
print("Aggregating dataset tasks completed!")
