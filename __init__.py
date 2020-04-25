import datetime as dt
import logging
import os
from typing import List, Tuple, Optional
from stocker import FundNotFoundException, add_indicators, Stock, StockServer

import matplotlib.pyplot as plt

logger = logging.getLogger()


def setup_loggers(args_parsed):
	""" Funkcija namesti izpisovanje v datoteko in na konzolo """
	uniquestr = dt.datetime.now().strftime("%d-%m-%Y_%H%M%S")
	dbg_log_formatter = logging.Formatter(
		fmt='%(asctime)-15s (%(relativeCreated)-8d ms) | %(levelname)-7s | %(threadName)-12.12s | %(name)15.15s | %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S')
	info_log_formatter = logging.Formatter(
		fmt='%(asctime)-15s | %(message)s',
		datefmt='2020-%b-%d %H:%M:%S')
	os.makedirs(args_parsed.log_dir, exist_ok=True)

	file_handler = logging.FileHandler(
		os.path.join(args_parsed.log_dir, 'file-%s.log' % uniquestr),
		mode='a')

	global logger
	logger.setLevel(logging.DEBUG)
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(dbg_log_formatter if args_parsed.log_level == logging.DEBUG else info_log_formatter)
	file_handler.setFormatter(dbg_log_formatter)
	console_handler.setLevel(args_parsed.log_level)
	file_handler.setLevel(logging.DEBUG)

	logger.addHandler(console_handler)
	logger.addHandler(file_handler)
	logger.debug(str(args_parsed))


def get_tickers(wws, *fund_tickers):
	# type: (StockServer, List[Tuple[str, Optional[str]]]) -> List[Stock]
	all_data = []
	for _ticker, _name in fund_tickers:
		try:
			all_data.append(wws[_ticker])
		except FundNotFoundException as fnfe:
			logging.exception(fnfe)
	return all_data


def display_command(wss, parsed_args):
	from stocker import stock_display
	stocks = {}

	for stock_query in parsed_args.stocks:
		stock = wss[stock_query]
		stocks[stock.ticker] = stock

	for k, stock in stocks.items():
		add_indicators(stock, parsed_args.markers)
		stock.calculate_total_investments()
		stock.write()
		stock_display.graph_stock(stock)
	plt.show()


def prediction_command(wss, parsed_args):
	from data_analysis import main as data_analysis_on
	seq_len = 48
	epochs = 50
	batch_size = 64
	stock = wss[parsed_args.stock]
	verbosity = 0 if parsed_args.log_level == logging.WARNING else 10
	data_analysis_on(stock, seq_len, epochs, batch_size, action=parsed_args.analysis_command, show=parsed_args.show_model, verbose=verbosity)


def main():
	from arguments import get_arg_parser
	parser = get_arg_parser()
	args_parsed = parser.parse_args()

	if args_parsed.command is None or args_parsed.command == 'help':
		parser.print_help()
		exit(0)
	setup_loggers(args_parsed)
	world_stock_server = StockServer()
	if args_parsed.command == 'display':
		display_command(world_stock_server, args_parsed)
	elif args_parsed.command == 'predict':
		prediction_command(world_stock_server, args_parsed)


if __name__ == '__main__':
	main()
