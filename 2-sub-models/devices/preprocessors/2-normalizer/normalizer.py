
for device in devices:
    device_cols = [col for col in bucketed_df.columns if col.startswith(f"{device}.")]
    bucketed_df[f"{device}.is_off"] = bucketed_df[device_cols].isnull().all(axis=1).astype(int)

bucketed_df['fridge.temp_condition'] = bucketed_df['fridge.temp_condition'].astype(str).str.strip().map({"high": 1, "low": 0})
bucketed_df['garage_door.door_state'] = bucketed_df['garage_door.door_state'].astype(str).str.strip().map({"open": 1, "closed": 0})
bucketed_df['garage_door.sphone_signal'] = bucketed_df['garage_door.sphone_signal'].astype(str).str.strip().map({"true": 1, "false": 0})
bucketed_df['motion_light.light_status'] = bucketed_df['motion_light.light_status'].astype(str).str.strip().map({"on": 1, "off": 0})

for col in bucketed_df.select_dtypes(include=['int64', 'float64']).columns:
    if col.endswith(".is_off"):
        continue
    device_name = col.split('.')[0]
    is_off_col = f"{device_name}.is_off"
    if is_off_col in bucketed_df.columns:
        off_mask = bucketed_df[is_off_col] == 1
        on_mask = ~off_mask
        active_min = bucketed_df.loc[on_mask, col].min()
        off_value = active_min - 1
        avg_value = bucketed_df.loc[on_mask, col].mean()
        if np.isnan(avg_value):
            avg_value = 0
        bucketed_df.loc[on_mask, col] = bucketed_df.loc[on_mask, col].fillna(avg_value)
        bucketed_df.loc[off_mask, col] = bucketed_df.loc[off_mask, col].fillna(off_value)
    else:
        avg_value = bucketed_df[col].mean()
        if np.isnan(avg_value):
            avg_value = 0
        bucketed_df[col] = bucketed_df[col].fillna(avg_value)

df_time_filled = backfill_time(bucketed_df)
for col in time_columns:
    device_name = col.split('.')[0]
    is_off_col = f"{device_name}.is_off"
    off_mask = bucketed_df[is_off_col] == 1
    on_mask = ~off_mask
    bucketed_df.loc[off_mask, col] = bucketed_df.loc[off_mask, col].fillna(baseline_date)
    bucketed_df.loc[on_mask, col] = df_time_filled.loc[on_mask, col]

for col in time_columns:
    bucketed_df[col] = pd.to_datetime(bucketed_df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')

base_timestamp = pd.Timestamp(baseline_date)

for col in time_columns:
    bucketed_df[col] = bucketed_df[col].apply(
        lambda x: (x - base_timestamp).total_seconds() if pd.notnull(x) else np.nan
    )

scaler = MinMaxScaler()
numeric_cols = list(bucketed_df.select_dtypes(include=['int64', 'float64']).columns)
bucketed_df[numeric_cols] = scaler.fit_transform(bucketed_df[numeric_cols])

cat_cols = [col for col in bucketed_df.select_dtypes(include="object").columns]
normalization_df = pd.get_dummies(bucketed_df, columns=cat_cols, drop_first=False)
normalization_df = normalization_df.replace({True: 1, False: 0})

print("normalizing dataset tasks completed!")
normalization_df.to_csv("normalized.csv", index=False)