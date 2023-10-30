import quantlib.data_utils as du
import quantlib.general_utils as gu
from dateutil.relativedelta import relativedelta
import random
import json
import pandas as pd

from subsystems.lbmom.subsys import Lbmom
# df, instruments = du.get_sp500_df()
# df = du.extend_dataframe(traded=instruments, df=df)
# gu.save_file("./Data/data.obj", (df, instruments))

df, instruments = gu.load_file("./Data/data.obj")
print(df, instruments)
# df = du.extend_dataframe(instruments, df)
# print(df)

# pairs = []

# while len(pairs) <= 20:
#     pair = random.sample(list(range(16, 300)), 2)
#     if pair[0] == pair[1]: continue
#     pairs.append((min(pair[0], pair[1]), max(pair[0], pair[1])))

# print(pairs)

# with open("./subsystems/lbmom/config.json", "w") as f:
#     json.dump({"instruments": instruments}, f, indent=4)

VOL_TARGET = 0.20

# run simulation for 5 years
print(df.index[-1])
sim_start = df.index[-1] - relativedelta(years=5)
print(sim_start)

strat = Lbmom(instruments_config="./subsystems/lbmom/config.json", historical_df=df, simulation_start=sim_start, vol_target=VOL_TARGET)

strat.get_subsys_pos()