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

label_columns = [
    'fridge.label', 'garage_door.label', 'gps_tracker.label',
    'modbus.label', 'motion_light.label', 'thermostat.label',
    'weather.label'
]

type_columns = [
    'fridge.type', 'garage_door.type', 'gps_tracker.type', 'modbus.type',
    'motion_light.type', 'thermostat.type', 'weather.type'
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
        get_first_feature(new, label_columns)[0] in extract_feature(pre, label_columns)[0] and
        get_first_feature(new, type_columns)[0] in extract_feature(pre, type_columns)[0] and
        get_first_feature(new, label_columns)[1] not in extract_feature(pre, label_columns)[1]
    ):
        return True
    return False

dataframes = []

for prefix, file in csv_files.items():
    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        format="%d-%b-%y %H:%M:%S",
        errors='raise'
    )
    df.drop(columns=["date", "time"], inplace=True)
    df = df.add_prefix(f"{prefix}.")
    new_datetime = f"{prefix}.datetime"
    cols = [new_datetime] + [col for col in df.columns if col != new_datetime]
    df = df[cols]
    dataframes.append(df)

combined_df = pd.concat(dataframes, axis=0, ignore_index=True)

for col in time_columns:
    combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')

combined_df["sort_timestamp"] = combined_df.apply(
    lambda row: get_first_feature(row, time_columns)[0], axis=1
)

sorted_df = combined_df.sort_values(by="sort_timestamp")
sorted_df.drop(columns=["sort_timestamp"], inplace=True)
sorted_df = sorted_df.reset_index(drop=True)

activity_logs = []

for idx, row in sorted_df.iterrows():
    if idx == 0:
        activity_log = row.to_dict()
        activity_logs.append(activity_log)
        continue
    previous_log = activity_logs[-1]
    if integration_checker(previous_log, row):
        for col in sorted_df.columns:
            if col not in previous_log or pd.isnull(previous_log[col]):
                previous_log[col] = row[col]
    else:
        previous_log["label"] = get_first_feature(previous_log, label_columns)[0]
        previous_log["type"]  = get_first_feature(previous_log, type_columns)[0]
        activity_log = row.to_dict()
        activity_logs.append(activity_log)

bucketed_df = pd.DataFrame(activity_logs)
bucketed_df.drop(columns=label_columns, inplace=True)
bucketed_df.drop(columns=type_columns, inplace=True)

for col in time_columns:
    bucketed_df[col] = bucketed_df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else x)

bucketed_df.to_csv("aggregated.csv", index=False)

print("Aggregating dataset tasks completed!")