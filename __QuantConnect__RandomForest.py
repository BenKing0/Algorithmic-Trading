import numpy as np
from RandomForrest import RandomForrest, percentage_change


class RandomForest(QCAlgorithm):

    def Initialize(self):

        self.entity = 'SPY'
        training_size = 1000
        backtest_period = 365
        self.including_short = False
        self.name = 'high'  # so far best was 'high'
        self.SetStartDate(datetime.now() - timedelta(backtest_period))
        self.SetEndDate(datetime.now())
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        self.previous = None

        try:
            ent = self.AddEquity(self.entity, Resolution.Daily)
        except:
            ent = self.AddForex(self.entity, Resolution.Daily)

        self.symbols = [ent.Symbol]
        # don't include backtest period in trianing set:
        self.htd = self.History(self.symbols[0], training_size + backtest_period, Resolution.Daily)[:training_size]
        self.rf = RandomForrest(self.htd)
        self.rf.train(n=5,
                      coi_name=self.name)  # Train on training_size samples taken before `live' (backtest) simulations

    def OnData(self, data):

        if self.previous is not None and self.previous.date() == self.Time.date():
            return

        holdings = self.Portfolio[self.symbols[0]].Quantity

        live_window = self.History(self.symbols[0], 5, Resolution.Daily)
        X = self.get_test_features(live_window)
        trade_signal = self.rf.test(override_features=X)

        self.Debug(f'Trade signal for {self.Time}: {trade_signal}')
        if holdings <= 0 and trade_signal == 1:  # ready to buy if predicting rise and not already invested
            self.SetHoldings(self.symbols[0], 1.0)
        elif holdings <= 0 and trade_signal == -1:  # ready to short if predicting drop and not already invested
            if self.including_short:
                self.SetHoldings(self.symbols[0], -1.0)
            else:
                pass
        elif holdings > 0 and trade_signal == -1:  # ready to liquidate if predicting drop and have already gone long
            self.Liquidate(self.symbols[0])
        self.Debug(f'Holdings now at {holdings}')

        self.previous = self.Time

    def get_test_features(self, window):

        open_, close, low, high = window['open'][-1], window['close'][-1], window['low'][-1], window['high'][-1]
        coi = list(window[self.name])
        open_close = (open_ - close) / open_
        high_low = (high - low) / low

        coi_percent_change = percentage_change(coi)
        mean = np.mean(coi_percent_change)
        std = np.std(coi_percent_change)

        X = [open_close, high_low, mean, std]

        return X