import backtrader as bt
from datetime import date

class PrintClose(bt.Strategy):

    def __init__(self):
        #Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def log(self, txt, dt=None):
        dt = dt 
        date = self.datas[0].datetime.date(0)
        print(f'{date}, {txt} {dt}') #Print date and close

    def next(self):
        self.log('Close: ', self.dataclose[0])
        
class MAcrossover(bt.Strategy): 
    # Moving average parameters
    params = (('pfast',10),('pslow',20),)

    def log(self, txt, dt=None):
        dt = dt 
        date = self.datas[0].datetime.date(0)
        print(f'{date}, {txt} {dt}') # Comment this line when running optimization

    def __init__(self):
        self.dataclose = self.datas[0].close
        
        # Order variable will contain ongoing order details/status - ie. currently in trade or order pending
        self.order = None

        # Instantiate moving averages - use built-in indicators to stay away from time-lag errors etc...
        self.slow_sma = bt.indicators.MovingAverageSimple(self.datas[0], 
                        period=self.params.pslow)
        self.fast_sma = bt.indicators.MovingAverageSimple(self.datas[0], 
                        period=self.params.pfast)
        
        # use built-in indicator crossover (if none available, can manually code the conditions in the self.next() section)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)
        
    def notify_order(self, order): # log when an order gets executed, and at what price
        if order.status in [order.Submitted, order.Accepted]:
            # An active Buy/Sell order has been submitted/accepted - Nothing to do
            return

        # Check if an order has been completed:
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Reset orders
        self.order = None
 
    # the logic of the trading - SMA in this example:
    def next(self):
        # Check for open orders
        if self.order:
            return

        # Check if we are in the market, as only want one open position at a time for this strategy
        if not self.position:
            # We are not in the market, look for a signal to OPEN trades

            # If the 20 SMA is above the 50 SMA
            
            if self.crossover > 0:
                self.log(f'BUY CREATE {self.dataclose[0]:2f}')
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
            # Otherwise if the 20 SMA is below the 50 SMA  
            elif self.crossover < 0:
                self.log(f'SELL CREATE {self.dataclose[0]:2f}')
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
        else:
            # We are already in the market, look for a signal to CLOSE trades
            if len(self) >= (self.bar_executed + 5):
                self.log(f'CLOSE CREATE {self.dataclose[0]:2f}')
                self.order = self.close()