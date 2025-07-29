import pandas as pd
from typing import List, Dict

def get_unique_categorical_values(df: pd.DataFrame, columns: List[str]) -> Dict[str, List[str]]:
    value_map = {}
    for col in columns:
        if col in df.columns:
            value_map[col] = sorted(df[col].dropna().unique().tolist())
        else:
            print(f">> Warning: Column not found in DataFrame: {col}")
    return value_map

def one_hot_encode_columns(
    df: pd.DataFrame,
    columns: List[str],
    category_map: Dict[str, List[str]],
    drop_original: bool = True
) -> pd.DataFrame:
    df_out = df.copy()
    for col in columns:
        if col not in df_out.columns:
            raise KeyError(f"Column not found in DataFrame: {col}")
        categories = category_map.get(col)
        if categories is None:
            raise ValueError(f"Missing category list for column: {col}")
        df_out[col] = pd.Categorical(df_out[col], categories=categories)
    dummies = pd.get_dummies(df_out[columns], prefix=columns, prefix_sep='.')
    dummies = dummies.replace({True: 1, False: 0}) 
    if drop_original:
        df_out.drop(columns=columns, inplace=True)
    return pd.concat([df_out, dummies], axis=1)

if __name__ == "__main__":
    df = pd.read_csv("../../../../../0-datasets & 1-hardware-estimate/netwrok/train_test_network.csv")
    cols_to_encode = ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_version", "ssl_cipher",
                      "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer", "http_trans_depth", "http_method", 
                      "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice", "http_version"]
    all_possible_values = get_unique_categorical_values(df, cols_to_encode)
    df = pd.read_csv("../4-numericUAgent/1-uAgentToCodebookIndex/uAgent2idx.csv")
    df_encoded = one_hot_encode_columns(df, cols_to_encode, all_possible_values)
    df_encoded.to_csv("cat2numeric.csv", index=False)
