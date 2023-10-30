import json 
import pandas as pd
import numpy as np
import quantlib.backtest_utils as backtest_utils
import quantlib.indicators_cal as indicators_cal

class Lbmom:
    def __init__(self, instruments_config, historical_df, simulation_start, vol_target):
        self.pairs = [(150, 192), (16, 271), (103, 176), (18, 200), (164, 245), (114, 131), (126, 298), (35, 260), (40, 91), (26, 190), (268, 272), (207, 275), (64, 146), (26, 129), (69, 251), (199, 293), (71, 140), (31, 70), (42, 166), (73, 75), (52, 161)]
        self.historical_df = historical_df
        self.simulation_start = simulation_start
        self.vol_target = vol_target

        # https://hangukquant.substack.com/p/volatility-targeting-the-strategy
        # https://hangukquant.substack.com/p/volatility-targeting-the-asset-level
        with open(instruments_config) as f:
            self.instruments_config = json.load(f)

        self.sysname = "LBMOM"

    def extend_historicals(self, instruments, historical_data):
        # we need to obtain data with regards to the LBMOM strategy
        # in particular we want the moving averages which is a proxy for momentum factor
        # we also want a univariate statistical factor as an indicator of regime. 
        # We use the average directional index ADX as a proxy for momentum regime indicator
        for inst in instruments:
            historical_data["{} adx".format(inst)] = indicators_cal.adx_series(
                high=historical_data["{} high".format(inst)],
                low=historical_data["{} low".format(inst)],
                close=historical_data["{} close".format(inst)],
                n=14
            )
            for pair in self.pairs:
                # calculate the fastMA - slowMA
                historical_data["{} ema{}".format(inst, str(pair))] = indicators_cal.ema_series(
                    historical_data["{} close".format(inst)], n=pair[0]) - indicators_cal.ema_series(
                        historical_data["{} close".format(inst)], n=pair[1])
                
        # the historical_data has all the info required for backtesting        
        return historical_data
        

    def run_simulation(self, historical_data):
        """
        Init Params
        """
        instruments = self.instruments_config["instruments"]

        """
        Pre-processing
        """
        # def is_halted(inst, date):
        #     print("is_haltea", inst, date)
        #     print(historical_data[:date].tail(3))

        #     return not np.isnan(historical_data.loc[date, "{} active".format(inst)]) and (~historical_data[:date].tail(3)["{} active".format(inst)]).any()
        
        historical_data = self.extend_historicals(instruments=instruments, historical_data=historical_data)
        #print(historical_data)
        portfolio_df = pd.DataFrame(index=historical_data[self.simulation_start:].index).reset_index()
        portfolio_df.loc[0, "capital"] = 10000
        is_halted = lambda inst, date: not np.isnan(historical_data.loc[date, "{} % active".format(inst)]) and (~historical_data[:date].tail(3)["{} % active".format(inst)].any())
        # this means that in order to not be a halted asset, it needs to have actively traded over the last 3 data points at the minimum
        #print(portfolio_df)
        #print(historical_data.columns)
        """
        Simulating
        We adopt a risk management technique at asset and strategy level called vol targeting.
        This in general means that we lever our captital to obtain a certain target annualized level of volatility, which is our proxy for risk /exposure.
        This is controlled by the parameter VOL_TARGET, that we pass from the main driver.
        The relative allocations in a vol target framework is that positions are inversely proportional to their volatility.
        In other words, a priori we assign the same risk to each position, when not taking into account the relative alpha (momentum) strengths

        So we assume 3 different risk/capital allocation techniques.
        1. Strategy vol targeting (vertical across time)
        2. Asset vol Targetting (relative across assets)
        3. Voting Systems (Indicating the degree of momentum factor)
        """
        for i in portfolio_df.index:
            date = portfolio_df.loc[i, "date"]
            strat_scalar = 2 # strategy scalar 
            nominal_total = 0
            
            tradable = [inst for inst in instruments if not is_halted(inst, date)]
            non_tradable = [inst for inst in instruments if inst not in tradable] 

            """
            Get PnL, Scalars
            """
            if i != 0:
                #pass
                date_prev = portfolio_df.loc[i-1, "date"]
                pnl, nominal_ret = backtest_utils.get_backtest_day_stats(portfolio_df, instruments, date, date_prev, i, historical_data)
                print(pnl, nominal_ret)
                #portfolio_df.loc[i, "capital"] = portfolio_df.loc[i - 1, "capital"] + pnl # already done in get_backtest_day_stats,

                # Obtain strategy scalar
                strat_scalar = backtest_utils.get_strat_scaler(portfolio_df, lookback=100, vol_target=self.vol_target, idx=i, default=strat_scalar)
                # the strategy leverage / scalar should dynamically equilibriate to achieve target exposure

            portfolio_df.loc[i, "strat scalar"] = strat_scalar

            """ Get Positions """
            for inst in non_tradable:
                # assign weight and position to 0 if not tradable
                # Considering 
                # 1. What is the market exposure of the entire strategy and what is the overal volatility derived from the strategy, called strategy scalar
                # 2. What is the relative risk of the asset compared to other assets at that date point, this is called the asset-asset scalar
                # 3. What is the degree of alpha that is expected to be present in the asset
                portfolio_df.loc[i, "{} units".format(inst)] = 0
                portfolio_df.loc[i, "{} w".format(inst)] = 0

            # to understand what is going on here, https://hangukquant.substack.com/p/volatility-targeting-the-asset-level
            
            for inst in tradable:
                # vote long if fastMa > slowMA else no vote. We trying to harvest momentum. We use MA pairs to proxy momentum, and define its strength by fraction of trending pairs.
                votes = [1 if (historical_data.loc[date, "{} ema{}".format(inst, str(pair))] > 0) else 0 for pair in self.pairs]
                #print(votes) # votes from the different MA crossover pairs
                forecast = np.sum(votes) / len(votes) # degree of momentum measured from 0 to 1, one if all trending, 0 if none trending
                forecast = 0 if historical_data.loc[date, "{} adx".format(inst)] < 25 else forecast # if regine is not trending, set forecast to 0

                position_vol_target = (1 / len(tradable)) * portfolio_df.loc[i, "capital"] * self.vol_target / np.sqrt(253) # dollar volatility assigned to a single position
                inst_price = historical_data.loc[date, "{} close".format(inst)]
                percent_ret_vol = historical_data.loc[date, "{} % ret vol".format(inst)] if historical_data.loc[:date].tail(20)["{} % active".format(inst)].all() else 0.025

                """
                percent_ret_vol = historical_data.loc[date, "{} % ret vol".format(inst)] if historical_data.loc[:date].tail(20)["{} active".format(inst)].all() else 0.025

                this means: if the asset has been actively traded in the past 20 days, then use rolling volatility as measure of asset vol, else default 0.025
                This is because, suppose an asset is not actively traded. Then its vol would be low, since there is little movement. Take an asset position inversely proportional
                to vol -> then the asset position would be large, since the reciprocal of vol is large. This would blow up the position sizing!
                """
                dollar_volatility = inst_price * percent_ret_vol # vol in nominal dollar terms
                position = strat_scalar * forecast * position_vol_target / dollar_volatility
                portfolio_df.loc[i, "{} units".format(inst)] = position
                #print(inst, position, forecast)
                nominal_total += abs(position * inst_price) #assuming no FX conversion is required

            for inst in tradable:
                units = portfolio_df.loc[i, "{} units".format(inst)]
                nominal_inst = units * historical_data.loc[date, "{} close".format(inst)]
                inst_w = nominal_inst / nominal_total
                portfolio_df.loc[i, "{} w".format(inst)] = inst_w


            """ Perform Calculations for Date """
            portfolio_df.loc[i, "nominal"] = nominal_total
            portfolio_df.loc[i, "leverage"] = nominal_total / portfolio_df.loc[i, "capital"]
            print(portfolio_df.loc[i])
        
        # store in excel file, to see what strategy generates
        portfolio_df.to_excel("libmom.xlsx")
        return portfolio_df, instruments

    def get_subsys_pos(self):
        portfolio_df, instruments = self.run_simulation(historical_data=self.historical_df)