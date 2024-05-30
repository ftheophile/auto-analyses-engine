# %matplotlib widget
import operator
import quantlib.data_utils as du
import quantlib.general_utils as gu
from dateutil.relativedelta import relativedelta
import random
import json
import os
import pandas as pd

import matplotlib.pyplot as plt
from datetime import date
from subsystems.lbmom.subsys import Lbmom
from subsystems.lsmom.subsys import Lsmom

def create_local_data(cnt):
    today = date.today()
    dfile = "./Data/"+str(today)+"-data.txt"
    if(not os.path.isfile(dfile)):
        df1, instruments = du.get_sp500_df(cnt)
        df = du.extend_dataframe(traded=instruments, df=df1, window_size=20, num_of_std=2)
        gu.save_file("./Data/data.obj", (df, instruments))
        #print(df, instruments)
        today = date.today()
        f = open(dfile, "w")
        f.write("Data as of "+str(today))
        f.close()
        return df1
    else: 
        pass

df1 = create_local_data(30)


def load_local_data():
    df, instruments = gu.load_file("./Data/data.obj")
    # df = du.extend_dataframe(instruments, df)
    #print(instruments)
    return df, instruments

df, instruments = load_local_data()

def generate_pairs(instruments, pairsize):
    pairs = []

    while len(pairs) < pairsize:
        pair = random.sample(list(range(16, 300)), 2)
        if pair[0] == pair[1]: continue
        pairs.append((min(pair[0], pair[1]), max(pair[0], pair[1])))

    #print(pairs)

    with open("./subsystems/lbmom/config.json", "w") as f:
        json.dump({"instruments": instruments}, f, indent=4)
    
    with open("./subsystems/lsmom/config.json", "w") as f:
        json.dump({"instruments": instruments}, f, indent=4)

    return pairs

pairs = generate_pairs(instruments, 20)

START_CAPITAL = 10000

def simulate_years(df, subsystem, pairs, yrs):

    VOL_TARGET = 0.20
    START_CAPITAL = 10000

    # run simulation for 5 years
    #print(df.index[-1])
    sim_start = df.index[-1] - relativedelta(years=yrs, months=2, days=0)
    print(subsystem, "Simulation start: ", sim_start)

    strat = None
    if subsystem == "lbmom":
        strat = Lbmom(instruments_config="./subsystems/lbmom/config.json", historical_df=df, simulation_start=sim_start, vol_target=VOL_TARGET)
    else:
        strat = Lsmom(instruments_config="./subsystems/lsmom/config.json", historical_df=df, simulation_start=sim_start, vol_target=VOL_TARGET)

    return strat.get_subsys_pos(pairs, START_CAPITAL)

#instruments= ['AAPL', 'ABT', 'AFL', 'AMC', 'AMD', 'AMZN', 'AOS', 'APD', 'BSX', 'CAT', 'CHD', 'EOG', 'GE', 'GOOGL', 'GSK', 'HLT', 'JPM', 'LDOS', 'LLY', 'LMT', 'META', 'MPWR', 'MSFT', 'NIO', 'NVDA', 'ORCL', 'PANW', 'PTC', 'SAP', 'SYK', 'TRGP', 'TSLA', 'VRCA', 'VRTX']
# pairs = [(25, 212), (190, 270), (154, 269), (127, 271), (139, 262), (160, 199), (111, 192), (190, 232), (32, 177), (190, 196), (172, 179), (93, 281), (46, 202), (130, 295), (143, 202), (72, 232), (130, 243), (113, 208), (45, 193), (105, 269)] # 38737 in  5 yrs
# pairs = [(87, 219), (88, 265), (75, 84), (170, 244), (153, 231), (32, 243), (95, 176), (51, 278), (37, 293), (110, 204), (91, 242), (214, 259), (49, 89), (206, 218), (206, 210), (114, 124), (206, 295), (101, 252), (216, 254), (184, 219)]  # 38150 in  5 yrs
# pairs = [(96, 250), (85, 132), (193, 216), (37, 180), (46, 71), (203, 218), (84, 256), (28, 216), (281, 284), (47, 252), (164, 296), (138, 284), (172, 244), (95, 190), (26, 190), (198, 288), (131, 243), (73, 251), (108, 173), (151, 273)]  # 37667 in  5 yrs
# pairs = [(104, 138), (146, 218), (162, 256), (155, 265), (49, 108), (55, 284), (141, 290), (41, 107), (92, 229), (63, 183), (171, 240), (216, 267), (196, 299), (142, 214), (46, 72), (101, 110), (46, 274), (154, 252), (96, 241), (198, 272)]  # 37545 in  5 yrs
# pairs = [(151, 273), (75, 89), (112, 221), (43, 147), (18, 205), (219, 266), (120, 290), (68, 185), (255, 263), (154, 241), (22, 268), (178, 268), (132, 145), (127, 163), (122, 164), (34, 249), (123, 226), (87, 228), (110, 232), (119, 147)] # 37403 in 5 yrs
# pairs = [(31, 191), (66, 251), (71, 220), (212, 265), (124, 142), (72, 174), (72, 295), (39, 51), (25, 239), (131, 166), (165, 171), (32, 227), (182, 221), (111, 287), (50, 199), (76, 227), (219, 250), (54, 234), (149, 289), (103, 246)] #  37100in 5 yrs
# pairs = [(69, 225), (40, 204), (114, 133), (153, 162), (126, 225), (74, 169), (106, 133), (72, 137), (194, 285), (100, 165), (197, 257), (36, 268), (175, 278), (126, 186), (181, 223), (74, 250), (32, 63), (206, 233), (32, 39), (48, 220)] # 13281  in  1 yrs
# pairs = [(144, 187), (49, 210), (110, 267), (129, 264), (19, 258), (202, 232), (123, 148), (46, 284), (197, 233), (153, 285), (153, 247), (64, 264), (74, 166), (122, 239), (74, 172), (90, 100), (24, 33), (63, 246), (141, 232), (217, 230)] # 13314 in  1 yrs

# pairs = [(157, 253), (208, 272), (233, 239), (59, 265), (130, 251), (61, 180), (121, 285), (90, 275), (60, 150), (76, 162), (33, 173), (217, 278), (99, 207), (188, 225), (117, 226), (36, 238), (101, 298), (80, 283), (39, 246), (211, 281)]  # 10907 in  0.5 yrs
# pairs = [(140, 201), (158, 268), (147, 251), (95, 176), (228, 270), (133, 189), (70, 224), (76, 154), (120, 138), (91, 157), (246, 286), (78, 95), (234, 293), (53, 238), (85, 129), (127, 141), (45, 296), (260, 280), (147, 213), (154, 249)] # 10910 in  0.5 yrs
# pairs = [(152, 224), (184, 245), (193, 272), (95, 176), (146, 235), (117, 243), (147, 266), (24, 281), (77, 233), (229, 294), (283, 293), (154, 270), (72, 208), (51, 75), (203, 226), (229, 270), (207, 296), (115, 165), (24, 230), (152, 251)]  # 10937 in  0.5 yrs
# pairs = [(79, 269), (145, 198), (255, 265), (37, 121), (122, 281), (130, 194), (194, 243), (145, 290), (268, 290), (114, 265), (222, 298), (109, 116), (106, 147), (122, 143), (39, 94), (197, 279), (55, 137), (200, 238), (156, 239), (61, 258)] # 10902 in  0.5 yrs
# pairs = [(140, 281), (173, 294), (167, 219), (105, 210), (52, 198), (64, 130), (67, 112), (70, 199), (193, 227), (168, 243), (182, 273), (22, 61), (117, 214), (52, 200), (151, 295), (191, 240), (126, 282), (103, 151), (111, 158), (48, 245)] # 10904 in  0.5 yrs
# pairs  = [(18, 61), (73, 289), (202, 214), (89, 147), (208, 280), (48, 62), (178, 286), (110, 296), (41, 264), (39, 289), (90, 253), (23, 283), (152, 237), (126, 217), (150, 183), (177, 247), (213, 238), (19, 288), (114, 171), (216, 246)]
yrs = 0

portfolio_lb, instruments_b = simulate_years(df=df, subsystem="lbmom", pairs=pairs, yrs=yrs)
# print(portfolio_lb)

# portfolio_ls, instruments_s = simulate_years(df=df, subsystem="lsmom", pairs=pairs, yrs=yrs)
# print(portfolio_ls)

#df1.head()
# Symbols with good results over 8 years
# ['SHW', 'TRGP', 'CCL', 'VICI', 'CHTR', 'EW', 'UAL', 'PODD', 'HUM', 'MDLZ', 'LLY', 'O', 'SBAC', 'MOH', 'CTLT', 'ZBRA', 'QCOM', 'NUE', 'ACN', 'BEN', 'ANSS', 'AAL', 'NOW', 'ADBE', 'LEN', 'RSG', 'ADSK', 'HCA', 'SJM', 'ATO', 'GOOGL', 'AMZN']

def daily_pnl_by_weekday(portfolio_df):
    daily_pnl = [sum(portfolio_df.loc[portfolio_df['weekday'] == i]["daily pnl"]) for i in range(1, 6)]
    print(daily_pnl)

def validate_results(portfolio_df, subsys):
    def filter_instruments(pair):
        _, value = pair
        #print(pair, value)
        
        #v = sum(value.values())
        if value < 0.0001:
            return False  # filter pair out of the dictionary
        else:
            return True  # keep pair in the filtered dictionary

    def display_details(dvalue):
        # units_df = portfolio_df.filter(like=' '+dvalue).join(portfolio_df["date"])
        # cols = units_df.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        # units_df = units_df[cols]  #    OR    df = df.ix[:, cols
        units_df = du.extract_columns_and_date(df=portfolio_df, dvalue=dvalue)
        # print(units_df.tail())
        filtered_units = dict(filter(filter_instruments, units_df.filter(like=' '+dvalue).iloc[-1].to_dict().items()))
        filtered_units2 = dict(filter(filter_instruments, units_df.filter(like=' '+dvalue).iloc[-2].to_dict().items()))
        #filtered_pnl = dict(filter(filter_instruments, portfolio_df.filter(like=' pnl').drop(['daily pnl'], axis=1).iloc[-1].to_dict().items()))
        
        
        print(units_df[filtered_units.keys()].tail())
        inst_pnl = portfolio_df.filter(like=' pnl').drop(['daily pnl'], axis=1).iloc[-1].to_dict()
        # print(inst_pnl)
        
        
        total = 0
        cprices = {}
        added_assets = []

        print("\n---------------- Portfolio ------------------")
        for col in filtered_units.keys():
            if col not in filtered_units2.keys():
                added_assets.append(col.split(" ")[0])
            if col != "date":
                col1 = col.split(" ")[0]
                inst_price = df[col1+' close'].iloc[-1]
                inst_units = filtered_units[col] # portfolio_df[col1+' units'].iloc[-1]
                total += inst_units * inst_price
                inst_ch_1d = inst_price - df[col1+' close'].iloc[-2]
                inst_ch_2d = df[col1+' close'].iloc[-2] - df[col1+' close'].iloc[-3]
                inst_ch_3d = df[col1+' close'].iloc[-3] - df[col1+' close'].iloc[-4]
                inst_ch_4d = df[col1+' close'].iloc[-4] - df[col1+' close'].iloc[-5]
                inst_ch1d_percent = (inst_ch_1d / df[col1+' close'].iloc[-2])*100
                y = lambda x: x > 0
                inst_strength = sum ([ y(x) for x in [inst_ch_1d, inst_ch_2d, inst_ch_3d, inst_ch_4d]])
                dollar2euro = lambda x: x * 0.92
                inst_price_euro = dollar2euro(inst_price)
                inst_potential = (6000 / inst_price_euro) *  dollar2euro(inst_ch_1d)

                inst_ch_4dg = inst_price - df[col1+' close'].iloc[-5]
                inst_pot_all = (6000 / inst_price_euro) *  dollar2euro(inst_ch_4dg)
                action = "{0:5}: {1:6.2f} {2:5} at {3:6.2f} ({4:6.2f}ℨ) with pnl {5:6.2f} ---- ch1 {6:6.2f}  ch2 {7:6.2f}  ch3 {8:6.2f}  ch4 {9:6.2f} and \%\ change {10:6.3f}  strength {11:2.0f}/4 pot_1d {12:7.2f}  pot_4d {13:7.2f}".format(col1, inst_units, dvalue, inst_price, inst_price_euro, inst_pnl[col1+' pnl'], inst_ch_1d, inst_ch_2d, inst_ch_3d, inst_ch_4d, inst_ch1d_percent, inst_strength, inst_potential, inst_pot_all)
                # print("%5s: %4.2f units at %4.2f % col1, inst_units, inst_price")
                print(action)
                # cprices.append(action)
                cprices[col1] = inst_ch1d_percent
        
        removed_assets = []
        for col in filtered_units2.keys():
            if col not in filtered_units.keys():
                removed_assets.append(col.split(" ")[0])
       
        print("\n------------- Portfolio changes ------------")
        print("Removed: ", removed_assets)
        print("  Added: ", added_assets)        
        # print(subsys, total, cprices)
        sortee = sorted(cprices.items(), key=lambda item: item[1],  reverse=True)
        print("\n\n--- Change Percentages ---\n", [i[0]+ "  {0:6.3f}".format(i[1]) for i in sortee][0:])


    print("###### Portfolio details for "+ subsys +" start ########")
    print("\nStart Capital: ", START_CAPITAL, "\nMax Capital:  ", max(portfolio_df["capital"]), "\nMin Capital:  ", min(portfolio_df["capital"]), "\nCapital:     ", portfolio_df["capital"].iloc[-1])
    capital_ret = (1 + portfolio_df["capital ret"]).cumprod()
    
    print(instruments)
    print(pairs, " in ", yrs, "yrs")
    daily_pnl_by_weekday(portfolio_df)
    print("\nValidator: ", portfolio_df["capital"].iloc[-1], capital_ret.iloc[-1], portfolio_df["capital"].iloc[-1]/START_CAPITAL - capital_ret.iloc[-1])
    if(portfolio_df["capital"].iloc[-1]/START_CAPITAL - capital_ret.iloc[-1] < 0.001):
        # print(portfolio_df.filter(like=' w').tail())
        # display_details('w')
        display_details('units')
        #print(sorted(portfolio_df.filter(like=' pnl').drop(['daily pnl'], axis=1).iloc[-1].to_dict().items(), key=operator.itemgetter(1)))
    else:
        print("Results not valid")
        print(portfolio_df["capital"])
        print(capital_ret)

    portfolio_df["capital"].plot()
    plt.show()

    #if(portfolio_df["capital"].iloc[-1]/START_CAPITAL - capital_ret.iloc[-1] < 0.001):
    #print(portfolio_df.filter(like=' w').tail())
    # units_df = portfolio_df.filter(like=' units').join(portfolio_df["date"])
    # cols = units_df.columns.tolist()
    # cols = cols[-1:] + cols[:-1]
    # units_df = units_df[cols]  #    OR    df = df.ix[:, cols
    # print(units_df.tail())
    # filtered_units = dict(filter(filter_instruments, units_df.filter(like=' units').tail(1).to_dict().items()))
    # print(subsys, filtered_units)

    # print("Results not valid")
    # print(portfolio_df["capital"])
    # print(capital_ret)

    print("\n###### Portfolio details for "+ subsys +" end  ########")

validate_results(portfolio_lb, "LBMOM")

def show_recent_ou_signals(df):
    def filter_overb(pair):
        _, value = pair
        #print(pair, value)
        
        #v = sum(value.values())
        if value != 1:
            return False  # filter pair out of the dictionary
        else:
            return True  # keep pair in the filtered dictionary

    def filter_underb(pair):
        _, value = pair
        #print(pair, value)
        
        #v = sum(value.values())
        if value != -1:
            return False  # filter pair out of the dictionary
        else:
            return True  # keep pair in the filtered dictionary

    inst_oub_df = df.filter(like=' '+"ou_signal")
    cols = inst_oub_df.columns.tolist()
    # units_df = du.extract_columns_and_date(df=df, dvalue=dvalue)  #    OR    df = df.ix[:, cols
    inst_oub_df = inst_oub_df[cols]
    # print(units_df.tail())
    def display_ou_results(inst_df, dvalue, daysback):

        overb_inst = dict(filter(filter_overb, inst_df.filter(like=' '+dvalue).iloc[daysback*(-1)-1].to_dict().items()))
        underb_inst = dict(filter(filter_underb, inst_df.filter(like=' '+dvalue).iloc[daysback*(-1)-1].to_dict().items()))

        over_bought = []
        under_bought = []
        for col in overb_inst.keys():
            over_bought.append(col.split(" ")[0])

        for col in underb_inst.keys():
            under_bought.append(col.split(" ")[0])

        
        print("   Days back :", daysback, "days")
        print(" Over Bought :", over_bought)
        print("Under Bought :", under_bought)
        print("\n")
        

    print("\n\n###### Market deviations start  ########")
    display_ou_results(inst_oub_df, "ou_signal", 0)
    display_ou_results(inst_oub_df, "ou_signal", 1)
    display_ou_results(inst_oub_df, "ou_signal", 2)
    display_ou_results(inst_oub_df, "ou_signal", 3)
    display_ou_results(inst_oub_df, "ou_signal", 4)
    print("###### Market deviations end  ########")

# validate_results(portfolio_ls, "LSMOM")
# cappital = []
# for i in range(10):
#     pairs = generate_pairs(instruments, 20)
#     portfolio_lb, instruments_b = simulate_years(df=df, subsystem="lbmom", pairs=pairs, yrs=yrs)
#     cappital.append(portfolio_lb["capital"].iloc[-1])
#     if(portfolio_lb["capital"].iloc[-1] > 13425):
#         print(pairs)
#     validate_results(portfolio_lb, str(i)+"-LBMOM")

# print(cappital)
show_recent_ou_signals(df) 
print("\a")


# https://neptune.ai/blog/predicting-stock-prices-using-machine-learning

