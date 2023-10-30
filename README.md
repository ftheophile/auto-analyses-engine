Use "prompt $G" to go in and out of shorten prompt in Windows DOS command prompt

Use "cd" to see current directory
Create python environment
python3 -m venv /path/to/new/virtual/environment
.\pyenv\Scripts\activate  


for ubuntu python venv
source upyenv/bin/activate

https://hangukquant.substack.com/p/1500-trading-strategy-deconstructed


# if the math was right, then our pnl and the equity curve should match

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("lbmom.xlsx")
df["capital"].plot()
plt.show() # that was our equity curve
df["capital"] # we started with 10k and ended with 21.5k

(1 + df["capital ret"]).cumprod() # it matches with our last cummulative product

# the value capital * cumproduct returns = ending capital should match if the math was correct
exit()
