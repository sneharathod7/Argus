import pandas as pd
df = pd.DataFrame({
    'j': ['A', 'A', 'B'], 
    't': pd.to_datetime(['2020-01-01', '2020-01-01 00:10', '2020-01-01']), 
    'v': [1,2,3]
})
r = df.groupby('j').rolling('45min', on='t')['v'].sum()
print(r)
print(r.index)
