import numpy as np

def convert_non_numerical(df):
    cols = df.columns.values
    for col in cols:
        text_digit_vals = {}
        def convert2int(val):
            return text_digit_vals[val]
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            col_values = set(df[col].values.tolist())
            i = 0
            for value in col_values:
                if value not in text_digit_vals:
                    text_digit_vals[value] = i
                    i += 1
            df[col] = list(map(convert2int, df[col]))
    return df