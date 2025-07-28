import pandas as pd
from typing import List

def one_hot_encode_columns(
    df: pd.DataFrame,
    columns: List[str],
    drop_original: bool = True
) -> pd.DataFrame:
    df_out = df.copy()
    missing = [c for c in columns if c not in df_out.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")
    dummies = pd.get_dummies(df_out[columns], prefix=columns, prefix_sep='.')
    dummies = pd.get_dummies(dummies, drop_first=False).replace({True: 1, False: 0})
    if drop_original:
        df_out.drop(columns=columns, inplace=True)
    return pd.concat([df_out, dummies], axis=1)


if __name__ == "__main__":
    df = pd.read_csv("../4-numericUAgent/1-uAgentToCodebookIndex/uAgent2idx.csv")
    cols_to_encode = ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_version", "ssl_cipher",
                      "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer", "http_trans_depth", "http_method", 
                      "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice", "http_version"]
    df_encoded = one_hot_encode_columns(df, cols_to_encode)
    df_encoded.to_csv("cat2numeric.csv", index=False)
