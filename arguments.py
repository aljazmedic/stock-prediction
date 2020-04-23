#!/usr/bin/python3
import argparse
import logging


def get_arg_parser():
	""" funkcija vrne objekt ArgumentParser, ki iz niza črk prebere argumente """
	parser = argparse.ArgumentParser(prog='stock', formatter_class=argparse.RawTextHelpFormatter)
	subprasers = parser.add_subparsers(title='command', dest='command')

	display_stock = subprasers.add_parser('display', help='displays candlebar chart of the stock')
	display_stock.add_argument('-s', '--stocks', nargs='+', dest='stocks', help='Stock tickers', default=[], required=True)
	display_stock.add_argument('-i', '--indicators', nargs='+', default=[],
							   help='indicators that are calculated: '
									'SMA.<n> - Simple moving average, '
									'EMA.<n> - Exponential moving average'
									'VWAP.<n> - Volume weighted average price', dest="markers", required=False)

	data_analysis_stock = subprasers.add_parser('predict', help='create model based on given stock')
	data_analysis_stock.add_argument('stock', help='stock ticker', default='AAPL')
	data_analysis_stock.add_argument('-d', '--display', action='store_true', dest='show_model', help='visualize model predictions', default=False)

	analysis_command = data_analysis_stock.add_mutually_exclusive_group()
	analysis_command.add_argument('-t', '--train',
								  action='store_const', const='train', dest='analysis_command',
								  help='train the model', default='load')
	analysis_command.add_argument('-l', '--load',
									 action='store_const', const='load', dest='analysis_command',
									 help='load the best checkpoint')
	dbg_level_group = parser.add_mutually_exclusive_group()
	dbg_level_group.add_argument(
		'-v', '--verbose',
		action="store_const",
		default=logging.INFO,
		const=logging.DEBUG,
		help="Verbose logging",
		dest="log_level")
	dbg_level_group.add_argument(
		'-q', '--quiet',
		action="store_const",
		const=logging.WARNING,
		help="No logging",
		dest="log_level")
	parser.add_argument('--log_dir', help="logs directory", dest="log_dir", type=str, default='logs/')

	return parser
