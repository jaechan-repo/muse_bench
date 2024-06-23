import pandas as pd
import numpy as np

p = 0.9
df = pd.read_csv("lyrics_popular_filtered.csv")
forget, retain = np.split(df, [int(p * len(df))])
forget.to_csv("lpf_forget.csv", index=False)
retain.to_csv("lpf_retain.csv", index=False)
