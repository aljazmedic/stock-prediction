import datetime as dt
import logging
import os
from typing import Union, Optional

import alpaca_trade_api as tradeapi
import dotenv

from stocker.stock import Stock
from stocker.stock_exceptions import FundNotFoundException, FundOutFileNameException

logger = logging.getLogger(__name__)
logging.getLogger("chardet").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


class StockServer:

	def __init__(self, file_path='world_stocks/'):
		self.default_file_location = self.set_default_outfile(file_path)
		dotenv.load_dotenv()
		self.api = tradeapi.REST(api_version='v2')
		acc = self.api.get_account()
		logger.debug(f"Alpaca account: {acc.status}")

	def _download_stock_data(self, fund_ticker, save_path, **kwargs):
		logger.info("Downloading data from Alpaca, stock '%s'" % fund_ticker)

		# And store date, low, high, volume, close, open values to a Pandas DataFrame
		alpaca_stock = self.api.get_barset(fund_ticker, 'day', limit=1000)
		if not alpaca_stock:
			raise FundNotFoundException("Invalid ticker: '%s'" % fund_ticker)
		_df = alpaca_stock.df[fund_ticker]
		_df.index.name = 'Date'
		logger.info("'%s' data saved to : %s" % (fund_ticker, save_path))
		_df.to_csv(
			save_path,
			sep=';',
			decimal=',',
			index_label='Date')
		return Stock(fund_ticker, _df, save_path)

	def __getitem__(self, query):
		# type: (str) -> Stock
		return self.get_stock(query)

	def set_default_outfile(self, path, file_name="%s-stock_market_data-%s.csv"):
		# type: (Union[bytes, str, os.PathLike], Optional[str]) -> Union[bytes, str, os.PathLike]
		if file_name.count("%s") != 2:
			raise FundOutFileNameException("Expected two '%s' parameters.")
		os.makedirs(path, exist_ok=True)
		self.default_file_location = os.path.join(path, file_name)
		return self.default_file_location

	def get_stock(self, _ticker):
		# type:(str) -> Stock
		logger.info("Getting stock info: %s" % _ticker)
		save_path = self.default_file_location % (_ticker, dt.datetime.now().strftime('%Y-%m-%d'))
		if os.path.exists(save_path):
			return Stock.from_csv(_ticker, save_path)
		else:
			return self._download_stock_data(_ticker, save_path)

	def __str__(self):
		return f"{type(self).__name__}()"


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from pandas.plotting import register_matplotlib_converters

	register_matplotlib_converters()
	world_stock_server = StockServer()

	plt.figure(figsize=(15, 7))

	ticker, name, df = world_stock_server.get_stock("AAPL")
	plt.plot(df.index, df["value"], label=name.replace('-', ' ').title())
	plt.xlabel('Date', fontsize=18)

	plt.legend()
	plt.show()
