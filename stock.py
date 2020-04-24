import datetime as dt
import logging
import os
from collections import defaultdict
from typing import Union, Dict, List, Any

import numpy as np
import pandas as pd
from matplotlib.dates import date2num

from decorators import *
from stock_exceptions import *

logger = logging.getLogger(__name__)
logging.getLogger("chardet").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connect").setLevel(logging.DEBUG)


class Stock(object):
	def __init__(self, ticker, dataframe, save_location):
		# type: (str, pd.DataFrame, Union[str, bytes]) -> None
		dataframe = dataframe.sort_index()  # .resample('B').mean()  # Create business frequency time data
		self.dataframe = dataframe
		self.ticker = ticker.upper()
		self.get_close_values()  # assure 'close'
		self.save_location = save_location

		self.groups = defaultdict(list)

		self.start_value = self.dataframe.loc[self.dataframe['close'].idxmin()]
		self.rsi(14)
		self.macd(9, slow=26, fast=12)

	@property
	def investments(self):
		# type: () -> Dict[List[Tuple[str, int, int, str]]]
		r = {}
		for inv in self.groups["investments"]:
			group_name = inv[3]
			r[group_name] = r.get(group_name, []) + [inv]
		return r

	@property
	def investments_list(self):
		# type: () -> List[Tuple[str, int, int, str]]
		return getattr(self, 'investments', [])

	@join_with_dataframe_as("all_investments", 'all_vep', 'all_input', default_make_unique=False)
	def calculate_total_investments(self):
		work_dataframe = self.dataframe[[]].copy()
		all_investments = []
		all_vep = []
		all_input = []
		for (investment_name, investment_vep, investment_input), _ in self.groups["investments"]:
			all_investments.append(investment_name)
			all_vep.append(investment_vep)
			all_input.append(investment_input)
		work_dataframe['all_investments'] = sum([self.dataframe[colname].fillna(0) for colname in all_investments])
		work_dataframe['all_vep'] = sum([self.dataframe[colname].fillna(0) for colname in all_investments])
		work_dataframe['all_input'] = sum([self.dataframe[colname].fillna(0) for colname in all_input])
		return work_dataframe

	def name_columns(self, name_c=None):
		# type: (Optional[callable]) -> set
		if name_c is None:
			def name_c(in_str):
				return "%s.%s" % (self.ticker, in_str)
		self.dataframe.rename(name_c, axis='columns')
		return self.dataframe.index

	def get_close_values(self):
		if "close" not in self.dataframe:
			self.dataframe["close"] = self.dataframe['value']
		return self.dataframe['close']

	@save_to_group('investments')
	@join_with_dataframe_as("investment", "investment_VEP", "investment_input")
	@default_perform_on('close')
	def _add_investment(self, *keys, date, st_vep, work_dataframe, col_idxs):
		# type: (dt.datetime, float, Optional, pd.DataFrame, Union[List, Tuple]) -> pd.DataFrame
		close_ = col_idxs
		start_index = work_dataframe.index.get_loc(date, method='bfill')
		work_dataframe["VEP"] = np.nan
		work_dataframe["input"] = np.nan
		work_dataframe["investition"] = np.nan
		work_dataframe.VEP[start_index:] = st_vep
		work_dataframe.investition[start_index:] = st_vep * work_dataframe.iloc[close_][start_index:]
		input_price = work_dataframe.investition[start_index]
		work_dataframe.input[start_index:] = input_price
		return work_dataframe[["investition", "VEP", "input"]]

	@ignore_stock_exceptions(StockInformationMissingException)
	@save_to_group("rsis")
	@join_with_dataframe_as("RS", "RSI", default_make_unique=False)
	@default_perform_on('close', 'open')
	def rsi(self, period, *, work_dataframe, col_idxs):  # Calculate RSI
		# type: (int, Optional, pd.DataFrame, Union[List, Tuple]) -> pd.DataFrame
		close_, open_ = col_idxs
		work_dataframe['change'] = work_dataframe.iloc[close_] - work_dataframe.iloc[
			open_]  # calculating gains and losses
		work_dataframe['gain'] = work_dataframe.change[work_dataframe.change > 0]  # new column of gains
		work_dataframe['loss'] = work_dataframe.change[work_dataframe.change < 0] * (-1)  # new column of losses
		work_dataframe.drop(columns=['change'], inplace=True)

		work_dataframe.gain.fillna(0, inplace=True)
		work_dataframe.loss.fillna(0, inplace=True)

		work_dataframe['again'] = work_dataframe.gain.rolling(
			period).mean()  # calculate the average gain in the last periods
		work_dataframe['aloss'] = work_dataframe.loss.rolling(
			period).mean()  # calculate the average loss in the last periods

		work_dataframe['RS'] = work_dataframe.again / work_dataframe.aloss  # calculating RS
		work_dataframe['RSI'] = 1 - (1 / (1 + work_dataframe.RS))  # calculating RSI
		work_dataframe.drop(columns=['gain', 'loss', 'again', 'aloss'], axis=1,
							inplace=True)  # remove undesired columns

		return work_dataframe

	@join_with_dataframe_as("K%")
	@default_perform_on("low", "high", "close")
	def add_k(self, period, *, work_dataframe, col_idxs):  # Calculate Stochastic Oscillator (%K)
		# type: (int, Optional, pd.DataFrame, Union[List, Tuple]) -> pd.DataFrame
		"""Return K indicator based on the dataframe"""
		low_, high_, close_ = col_idxs
		work_dataframe['L14'] = work_dataframe.iloc[low_].rolling(
			period).min()  # find the lowest price in the last 14 periods
		work_dataframe['H14'] = work_dataframe.iloc[high_].rolling(
			period).max()  # find the highest price in the last 14 periods
		work_dataframe['K%'] = (
				(work_dataframe.iloc[close_] - work_dataframe.L14) / (work_dataframe.H14 - work_dataframe.L14))
		work_dataframe.drop(columns=['L14', 'H14'], inplace=True)  # remove columns L14 and H14

		return work_dataframe

	@join_with_dataframe_as("R%")
	@default_perform_on("low", "high", "close")
	def add_r(self, period, *, work_dataframe, col_idxs):  # Calculate Larry William indicator (%R)
		# type: (int, Optional, pd.DataFrame, Union[List, Tuple]) -> pd.DataFrame
		"""Return R indicator based on the dataframe"""
		low_, high_, close_ = col_idxs
		work_dataframe['LL'] = work_dataframe.iloc[low_].rolling(
			period).min()  # find the lowest low price in the last 14 periods
		work_dataframe['HH'] = work_dataframe.iloc[high_].rolling(
			period).max()  # find the highest high price in the last 14 periods
		work_dataframe["R%"] = ((work_dataframe.HH - work_dataframe.iloc[close_]) / (
				work_dataframe.HH - work_dataframe.LL)) * (-1)
		work_dataframe.drop(columns=['HH', 'LL'], inplace=True)  # remove columns HH and LL

		return work_dataframe

	@join_with_dataframe_as("YC")
	@default_perform_on('close')
	def ycp(self, *, work_dataframe, col_idxs):
		# type: (Optional, pd.DataFrame, Union[List, Tuple]) -> pd.DataFrame
		""" Return yield change"""
		close_ = col_idxs
		start_val_iloc_index = work_dataframe.idxmin(), close_[
			1]  # first date and column in (slice(None, None, None), col)
		work_dataframe["yield_change"] = np.round(
			((work_dataframe.iloc[close_] / work_dataframe.iloc[start_val_iloc_index]) - 1), 2)
		return work_dataframe

	@join_with_dataframe_as("DYP")
	@default_perform_on('yield_change')
	def dyp(self, *, work_dataframe, col_idxs):
		# type: (Optional, pd.DataFrame, Union[List, Tuple]) -> pd.DataFrame
		"""Returns daily yield percent"""
		yc_ = col_idxs
		work_dataframe["dyp"] = np.round(work_dataframe.diff(periods=1).iloc[yc_] / 100, 2)
		return work_dataframe

	@save_to_group("smas")
	@join_with_dataframe_as("SMA")
	@default_perform_on('close')
	def sma(self, window, *, work_dataframe, col_idxs):  # Simple moving average
		# type: (int, Optional, pd.DataFrame, Union[List, Tuple]) -> pd.DataFrame
		"""Returns simple moving average"""
		close_ = col_idxs
		work_dataframe["sma"] = work_dataframe.iloc[close_].rolling(window).mean()  # Rolling of the first column
		return work_dataframe

	@save_to_group("emas")
	@join_with_dataframe_as("EMA")
	@default_perform_on('close')
	def ema(self, window, *, work_dataframe, col_idxs):  # Exponential moving average, perform on -> custom col
		# type: (int, Optional, pd.DataFrame, Union[List, Tuple]) -> pd.DataFrame
		"""Returns the exponential moving average"""
		close_ = col_idxs
		weights = np.exp(np.linspace(-1., 0., window))
		weights /= weights.sum()
		work_dataframe["ema"] = np.convolve(work_dataframe.iloc[close_], weights, 'full')[
								:len(work_dataframe.iloc[close_])]
		work_dataframe["ema"][:window] = np.nan  # work_dataframe["ema"][window]
		return work_dataframe

	@ignore_stock_exceptions(StockInformationMissingException)
	@join_with_dataframe_as("MACD", "MACD_signal", "MACD_histogram", default_make_unique=False)
	@default_perform_on("close")
	def macd(self, histogram, slow=26, fast=12, *, work_dataframe, col_idxs):
		# type: (int, Optional[int], Optional[int], Optional, pd.DataFrame, List) -> pd.DataFrame
		"""Returns MACD"""
		"""
		Instructions
		macd line = 12ema - 26 ema ## fast - slow
		signal line = 9ema of macd
		histogram = macd - signal
		"""
		close_ = col_idxs
		ema_slow = self.ema(slow, work_dataframe=work_dataframe[['close']], work_columns=("close",),
							column_names=("EMA_Slow",), make_unique_name=False,
							group_name=None)
		ema_fast = self.ema(fast, work_dataframe=work_dataframe[['close']], work_columns=("close",),
							column_names=("EMA_Fast",), make_unique_name=False,
							group_name=None)
		work_dataframe = work_dataframe.join([ema_fast, ema_slow])
		work_dataframe['MACD'] = work_dataframe["EMA_Fast"] - work_dataframe["EMA_Slow"]

		signal_df = self.ema(histogram, work_dataframe=work_dataframe[['MACD']], work_columns=("MACD",),
							 column_names=("MACD_signal",), make_unique_name=False,
							 group_name=None)
		work_dataframe = work_dataframe.join(signal_df)
		work_dataframe["MACD_histogram"] = work_dataframe.MACD - work_dataframe.MACD_signal
		work_dataframe.drop(columns=["close", "EMA_Slow", "EMA_Fast"], axis=1, inplace=True)
		return work_dataframe

	@ignore_stock_exceptions(StockInformationMissingException)
	@save_to_group("vwaps")
	@join_with_dataframe_as("VWAP")
	@default_perform_on("high", "low", "volume")
	def vwap(self, window, *, work_dataframe, col_idxs):  # Volume weighted average
		# type: (int, Optional, pd.DataFrame, List) -> pd.DataFrame
		"""Calculates trailing Volume weighted average price"""
		if any(_col not in self.dataframe for _col in ['high', 'low', 'close', 'volume']):
			raise StockInformationMissingException(
				"No high, low, close columns found in: '%s'" % "', '".join(self.dataframe.columns))
		high_, low_, volume_ = col_idxs
		work_dataframe["volume-scaled-price"] = \
			(work_dataframe.iloc[high_] + work_dataframe.iloc[low_]) / 2 * work_dataframe.iloc[volume_]
		work_dataframe["vwap"] = work_dataframe["volume-scaled-price"].rolling(
			window).sum() / work_dataframe.iloc[volume_].rolling(window).sum()
		work_dataframe.drop(columns=["volume-scaled-price"], inplace=True)
		return work_dataframe

	def ohlc(self):
		if any(_col not in self.dataframe for _col in ['open', 'high', 'low', 'close']):
			raise StockCandleInformatinMissingException(
				"No OHLC columns found in: '%s'" % "', '".join(self.dataframe.columns))
		plot_df = self.dataframe[['open', 'high', 'low', 'close']].copy()
		plot_df.reset_index(inplace=True)

		plot_df['Date'] = pd.to_datetime(plot_df['Date'], utc=True).map(date2num)
		return plot_df.loc[:, ['Date', 'open', 'high', 'low', 'close']]

	def update(self, other):
		# type: (Union[Stock, pd.DataFrame]) -> Stock
		if isinstance(other, Stock):
			self.dataframe = self.dataframe.update(other.dataframe)
		else:
			self.dataframe = self.dataframe.update(other)
		return self

	def __iter__(self):
		return iter(self.dataframe)

	def __str__(self):
		return "Stock('%s', (%s, %s))" % (
			self.ticker, str(self.dataframe.index[0]), str(self.dataframe.index[-1])
		)

	def __eq__(self, other):
		if isinstance(other, Stock):
			return self.ticker == other.ticker
		return False

	def unpack(self, dataframe_trim_funciton=lambda x: x):
		return self.ticker, dataframe_trim_funciton(self.dataframe)

	def __getitem__(self, item):
		if item not in self.dataframe:
			raise StockInformationMissingException
		return self.dataframe.__getitem__(item)

	def __setitem__(self, key, value):
		return self.dataframe.__setitem__(key, value)

	def write(self, file_location=None):
		if file_location is None:
			file_location = self.save_location
		self.dataframe.to_csv(
			file_location,
			sep=';',
			decimal=',',
			index_label='Date'
		)

	def extract_dataframe(self, *subcalls, cols=None):
		# type: (Tuple[SubCall], Optional[Tuple]) -> pd.DataFrame
		"""It returns copy of stocks dataframe with given columns of interest."""
		if cols is None:
			cols = []
		else:
			cols = list(cols)
		work_dataframe = self.dataframe[cols].copy()
		work_dataframe = work_dataframe.join(
			[subcall.call(self, group_name=None, no_join=True) for subcall in subcalls])
		return work_dataframe

	def prediction_model_data(self, dropna=True):
		# type: (Optional[bool]) -> pd.DataFrame
		log_diff_periods = 1
		work_dataframe = self.extract_dataframe(
			cols=("close", "open", "volume")
		)
		work_dataframe['Date'] = work_dataframe.index
		# work_dataframe["log_mid"] = np.log((work_dataframe.close + work_dataframe.open) / 2)  # sample [n]
		work_dataframe["log_mid"] = np.log(work_dataframe.close)  # sample [n]
		work_dataframe["log_mid_delta_preceding"] = work_dataframe["log_mid"].diff(log_diff_periods).shift(
			-log_diff_periods)
		work_dataframe['target'] = ((work_dataframe["log_mid_delta_preceding"] >= 0).astype(np.float32)).where(
			work_dataframe["log_mid_delta_preceding"].notna())
		# (np.log(work_dataframe["VWAP_preceding"]) - np.log(work_dataframe["SMA_trailing"])) > 0  # target [n]
		work_dataframe['RIC'] = self.ticker

		work_dataframe = work_dataframe[["log_mid_delta_preceding", "RIC", "target", "close", 'Date']] \
			.sort_index()
		if dropna:
			work_dataframe.dropna(inplace=True)
		work_dataframe.reset_index(drop=True, inplace=True)
		# work_dataframe.index.name = "index"
		# print(work_dataframe.isnull().sum())
		# print(work_dataframe.shape)
		return work_dataframe

	def add_csv_invsetments(self, path_to_csv, date_column, vep_column, label_options=None, filter_func=None):
		# type: (Union[str, bytes, os.PathLike], str, str, Optional[list], Optional[callable]) -> None
		logger.debug("%s: Adding investments from csv from '%s'" % (self.ticker, str(path_to_csv)))
		if filter_func is None:
			def filter_func(_):
				return True
		investments_dataframe = pd.read_csv(path_to_csv, delimiter=';', decimal=',', parse_dates=[date_column])
		no_of_investments = 0
		for index, row in investments_dataframe.iterrows():
			if not filter_func(row):
				continue
			no_of_investments += 1
			distinctables_list = [str(row[col]).lower().replace('č', 'c').replace('š', 's').replace('ž', 'z') for col in
								  label_options] + [str(no_of_investments)]
			self._add_investment(
				date=row[date_column],  # Leave kwargs, so only args (=Nothing) get passed to naming unique col
				st_vep=row[vep_column],
				*distinctables_list  # ,
				# rsuffix="_".join(distinctables_list)
			)
		logger.info(f"Added {no_of_investments} investment entries")

	@classmethod
	def from_csv(cls, fund_ticker, save_path):
		logger.info("Stock '%s' file already exists. Loading data from CSV" % fund_ticker)
		_df = pd.read_csv(
			save_path,
			sep=';',
			decimal=',',
			parse_dates=['Date'],
			index_col=['Date'])
		return cls(fund_ticker, _df, save_path)


class SubCall:
	""" Objekt, ki na funkciji pokliče metodo z danimi argumenti """

	def __init__(self, name, *args, **kwargs):
		self.name = name
		self.args = args
		self.kwargs = kwargs

	def call(self, stock, **force):
		# type:(Stock, Dict) -> Any
		func = getattr(stock, self.name, None)
		if func is not None:
			self.kwargs.update(force)
			return func(*self.args, **self.kwargs)
		raise Exception("Invalid attribute: " + self.name)


def add_indicators(stock, indicators):
	# type: (Stock, List[str]) -> Stock
	""" Funkcija doda indikatorje na podlagi nizov """
	for indicator_str in indicators:
		try:
			sub_call = _parse_function(indicator_str)
			sub_call.call(stock)
		except Exception as e:
			logger.exception(e)
	return stock


def _parse_function(in_str):
	# type: (str) -> SubCall
	first_dot_index = in_str.index('.')
	name, arguments = in_str[:first_dot_index], in_str[first_dot_index + 1:]
	args = [int(e.strip()) for e in arguments.split(".")]
	name = name.lower()
	if name in ['sma', 'ema', 'vwap']:
		return SubCall(name, *args)
	raise Exception(f"Indicator '{name}' unknown")


if __name__ == '__main__':
	from stock_server import StockServer

	world_stock_server = StockServer()
	MSFT_stock = world_stock_server["MSFT"]
	model_data = MSFT_stock.prediction_model_data()
	print(model_data.head(10), "... ...", model_data.tail(15), sep="\n")

	print(model_data.columns)
	print(MSFT_stock.dataframe.shape, "->", model_data.shape)
