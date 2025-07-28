import pandas as pd

def analyze_csv_columns(file_path, threshold=20):
    df = pd.read_csv(file_path)
    results = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            unique_vals = df[col].nunique(dropna=True)
            if unique_vals <= threshold:
                strategy = "One-Hot Encoding"
            else:
                strategy = "DVQ-VAE (High cardinality)"
            results[col] = {
                "type": "string",
                "unique_values": unique_vals,
                "strategy": strategy
            }
        else:
            results[col] = {
                "type": str(dtype),
                "strategy": "Numerical or other"
            }
    return results

results = analyze_csv_columns("../../0-testTrainSplitter/train_split.csv")
for res in results:
    print(f"{res}: ${results[res]}")