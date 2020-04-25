from itertools import cycle
from typing import Dict, List, Optional

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_finance import candlestick_ohlc
from pandas.plotting import register_matplotlib_converters

from stocker import decorators
import logging
from stocker.stock_exceptions import StockInformationMissingException
from stocker.stock import Stock

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

matplotlib.rcParams.update({
	'font.size': 9,
	'font.sans-serif': 'Arial',
	'font.family': 'sans-serif',
	'text.color': '#909090',
	'axes.labelcolor': '#909090',
	'xtick.color': 'w',
	'ytick.color': 'w',
	'grid.color': 'w',
	'axes.facecolor': '#07000d',
	"savefig.dpi": 1000
})
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 10000)


def plot_stock_price(ax, stock, colors, only_close=False):
	# type: (matplotlib.axes.Axes, Stock, Dict, Optional[bool]) -> None
	ticker, df = stock.unpack()
	try:
		if only_close:
			raise StockInformationMissingException()
		ohlc = stock.ohlc()
		candlestick_ohlc(
			ax,
			ohlc.values,
			width=0.5,
			colorup=colors['colorup'],
			colordown=colors['colordown'],
			# colordown_line=colors['shadow_down'],
			# colorup_line=colors['shadow_up'],
			alpha=0.8)
	except StockInformationMissingException:
		ax.plot(df.index, df['close'], color='#03e8fc')
	# ax.yaxis.set_color(colors['ylabels'])
	# ax.tick_params(axis='y')
	# ax.tick_params(axis='x')
	ax.set_ylabel('Stock Price and Volume')


def plot_stock_groups(ax, stock, colors, *groups):
	# type: (matplotlib.axes.Axes, Stock, Dict, List[str]) -> None
	ticker, df = stock.unpack()
	ax.grid(True)
	for g in groups:
		for group, _ in stock.groups[g]:
			group_name = group[0]
			ax.plot(
				df.index,
				df[group_name],
				label=group_name.replace("_", " "),
				# color=next(colors['AVGs']),
				linewidth=1.6,
				alpha=0.6)


@decorators.ignore_stock_exceptions(StockInformationMissingException)
def plot_rsi(ax, stock, colors):
	# type: (matplotlib.axes.Axes, Stock, Dict) -> None
	ticker, df = stock.unpack()
	if 'RSI' not in df:
		return
	ax.plot(df.index, df['RSI'], color=colors['RSI'])
	ax.set_ylim(0, 1)
	ax.axhline(.7, color=colors['RSI'], alpha=.6)
	ax.axhline(.3, color=colors['RSI'], alpha=.6)
	ax.fill_between(df.index, df['RSI'], .7, where=df['RSI'] >= .7, facecolor=colors['RSI_70'], alpha=.6)
	ax.fill_between(df.index, df['RSI'], .3, where=df['RSI'] <= .3, facecolor=colors['RSI_30'], alpha=.6)
	ax.tick_params(axis='y')
	ax.yaxis.set_major_locator(mticker.MaxNLocator(prune='lower'))
	ax.set_yticks([0., .3, .7, 1.])
	ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
	ax.set_ylabel('RSI')


@decorators.ignore_stock_exceptions(StockInformationMissingException)
def plot_volume(ax, stock, colors):
	# type: (matplotlib.axes.Axes, Stock, Dict) -> None
	ticker, df = stock.unpack()
	if 'volume' not in df:
		return
	ax.fill_between(df.index, 0, df['volume'], facecolor=colors['volume'], alpha=.5)
	ax.set_ylim(0, 2 * df['volume'].max())

	ax.yaxis.set_ticklabels([])
	ax.grid(False)
	plt.setp(ax.get_yticklabels(), visible=False)


def plot_macd(ax, stock, colors):
	# type: (matplotlib.axes.Axes, Stock, Dict) -> None
	ticker, df = stock.unpack()
	ax.tick_params(axis='x')
	ax.plot(df.index, df['MACD'])

	ax.plot(df.index, df['MACD_signal'], color='orange')
	# ax.fill_between(df.index, df['MACD_histogram'], 0, facecolor=colors['MACD'], alpha=.5)
	ax.bar(df.index, df['MACD_histogram'].where(df['MACD_histogram'] > 0), facecolor='green', alpha=.5)
	ax.bar(df.index, df['MACD_histogram'].where(df['MACD_histogram'] < 0), facecolor='red', alpha=.5)
	max_macd = max(
		df['MACD'].max(), -df['MACD'].min(),
		(df['MACD_histogram']).max(),
		-(df['MACD_histogram']).min())
	ax.yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
	ax.set_ylim(-max_macd, max_macd)

	ax.set_ylabel('MACD')


def plot_predictions(ax, prediction_data):
	# type: (matplotlib.axes.Axes, List[List]) -> None
	from matplotlib.lines import Line2D
	from matplotlib.patches import Patch
	for (pred_time, close_price, pred, act) in zip(*prediction_data):
		marker = 10 if pred == 1 else 11
		if np.isnan(act):
			color = 'white'
		else:
			color = 'r' if ((act > 0) ^ (pred == 1)) else 'g'
		ax.plot([pred_time], [close_price], c=color, marker=marker, markersize=7)

	legend_entries = [
		Line2D([0], [0], linestyle=None, color='white', label='Predvideno padanje', marker=11, markersize=7),
		Line2D([0], [0], color='white', label='Predvideno naraščanje', marker=10, markersize=7),
		Patch(color='g', label='Pravilno'),
		Patch(color='r', label='Napačno')
	]
	ax.legend(handles=legend_entries, loc='best')

	# ax.scatter([seq_len+1], [one_predicted], color='g')


def graph_stock(stock, predictions=None, title=None):
	register_matplotlib_converters()
	ticker, df = stock.unpack()
	colors = {
		'colorup': '#9eff15',
		'colordown': '#ff1717',
		'shadow_up': 'w',
		'shadow_down': 'w',
		'spine_color': '#5998ff',
		'volume': '#00ffe8',
		'MACD': '#00ffe8',
		'RSI': '#00ffe8',
		'RSI_70': '#fd5f6b',
		'RSI_30': '#6bfd5f',
		'SMA': cycle(['#5998ff', '#ff5998', '#98ff59']),
		'AVGs': cycle(('#abfef8', '#abdbfe', '#abfecf')),
	}
	fig = plt.figure(figsize=(12, 7), facecolor=matplotlib.rcParams['axes.facecolor'])
	axs = {1: plt.subplot2grid((6, 1), (1, 0), rowspan=4, colspan=1)}
	axs[0] = plt.subplot2grid((6, 1), (0, 0), sharex=axs[1], rowspan=1, colspan=1)  # RSI
	axs[2] = plt.subplot2grid((6, 1), (5, 0), sharex=axs[1], rowspan=1, colspan=1)  # MACD
	bottom_ax = axs[2]

	# ax1.set_title('%s Share Price' % ticker, color='white')
	plot_stock_groups(axs[1], stock, colors, 'smas', 'macds', 'vwaps')
	plot_stock_price(axs[1], stock, colors=colors, only_close=predictions is not None)
	plot_rsi(axs[0], stock, colors=colors)
	if 'volume' in stock:
		axs['1v'] = axs[1].twinx()  # Volume
		plot_volume(axs['1v'], stock, colors=colors)

	plot_macd(axs[2], stock, colors=colors)

	if predictions is not None:
		plot_predictions(axs[1], predictions)

	# TODO Fix plot_investments, create naming based on label
	for _, ax in axs.items():
		ax.spines['bottom'].set_color(colors['spine_color'])
		ax.spines['top'].set_color(colors['spine_color'])
		ax.spines['left'].set_color(colors['spine_color'])
		ax.spines['right'].set_color(colors['spine_color'])
		if ax is not bottom_ax:
			plt.setp(ax.get_xticklabels(), visible=False)

	bottom_ax.tick_params(axis='x')
	# plt.setp(bottom_ax.get_xticklabels(), visible=True)
	# plt.ylabel('Volume', color=colors['ylabels'])

	plt.suptitle("%s" % stock.ticker.upper())
	plt.subplots_adjust(hspace=0)

	handles, labels = axs[1].get_legend_handles_labels()

	if handles:
		axs[1].legend(handles, labels,
			loc=9,
			ncol=2,
			fancybox=True,
			prop={'size': 7},
			borderaxespad=0.)

	fig.suptitle(ticker.upper() if title is None else title, fontsize=11)
	fig.canvas.set_window_title(ticker.replace('-', ' ').title())

	fig.savefig('last_chart.png', facecolor=fig.get_facecolor())
	return fig
