from functools import wraps
import logging
from stocker.stock_exceptions import StockInformationMissingException
from typing import Tuple, Optional, Type

""" Dektoratorji za metode razreda Stock """
logger = logging.getLogger(__name__)


def error_call(func_to_call, error_to_raise):
	# type: (callable, Type[StockInformationMissingException]) -> Optional
	def _ignore(_e):
		logger.info(f"Ignored exception {_e}")

	def _raise(_e):
		raise _e

	d = {
		"ignore": _ignore,
		"raise": _raise
	}
	d.get(func_to_call, lambda x: x())(error_to_raise)


def ignore_stock_exceptions(*ignore_exceptions):
	# type: (Tuple[str]) -> callable
	""" Ignores given stock exceptions """

	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			try:
				return func(*args, **kwargs)
			except BaseException as be:
				if type(be) in ignore_exceptions:
					return
				raise be

		return wrapper

	return decorator


def debug(func):
	""" Debug helper function, logs inputs and outputs of functions"""
	def get_string(o):
		""" represents object with shortest notation """
		ret_s = str(o)
		ret_s = ret_s.replace("\n", "")
		if type(o) == list:
			return f"[{','.join([get_string(elem) for elem in o])}]"
		elif type(o) == tuple:
			return f"({','.join([get_string(elem) for elem in o])})"
		if len(ret_s) >= 15:
			return str(type(o))
		return ret_s

	@wraps(func)
	def wrapper(*args, **kwargs):
		args_repr = [get_string(o) for o in args]
		kwarg_repr = [f"{k}={get_string(v)}" for k, v in kwargs.items()]
		logger.debug(" ".join(
			("Calling function", func.__name__, f"with args   ({', '.join(args_repr)})")
		))
		logger.debug(" ".join(
			("                ", " "*len(func.__name__), f"with kwargs ({', '.join(kwarg_repr)})")
		))
		_ret = func(*args, **kwargs)
		logger.debug(" ".join(
			("Returned by     ", func.__name__, f" ({get_string(_ret)})")
		))
		return _ret

	return wrapper


def default_perform_on(*default_work_columns, default_errors='raise'):
	# type: (Tuple[str], Optional[str]) -> callable
	""" Gives function an option to perform other columns """
	column_indexes = [(slice(None, None), i) for i in range(len(default_work_columns))]
	if len(column_indexes) == 1:
		# If there is only one value, it doesnt have to be unpacked, rather forwarded
		column_indexes = column_indexes[0]

	def decorator(func):
		@wraps(func)
		def wrapper(*args, work_dataframe=None, work_columns=default_work_columns, errors=default_errors, **kwargs):
			_work_columns = [work_columns] if isinstance(work_columns, str) else list(
				work_columns)  # makes sure there is a list
			if work_dataframe is None:
				if any([_wc not in args[0].dataframe for _wc in _work_columns]):
					r = error_call(errors, StockInformationMissingException)
					if r is None:
						return
				_work_dataframe = args[0].dataframe[_work_columns].copy()
			else:
				_work_dataframe = work_dataframe.copy()
				if any([_wc not in _work_dataframe for _wc in _work_columns]):
					error_call(errors, StockInformationMissingException)
				_work_columns = _work_dataframe.columns.to_list()
			_ret_dataframe = func(*args, work_dataframe=_work_dataframe, col_idxs=column_indexes, **kwargs)
			_ret_dataframe.drop(columns=_work_columns, errors='ignore', inplace=True)
			return _ret_dataframe

		return wrapper
	return decorator


def join_with_dataframe_as(*default_column_names, default_make_unique=True, default_rsuffix=None):
	# type: (Optional[Tuple[str]], Optional[bool], Optional[str]) -> callable
	""" Saves the output series to stock column """

	def decorator(func):
		def prepare_names(names_in, args, make_unique=True, rsuffix=None):
			""" Assures that correctly formatted string tuple is returned """
			if isinstance(names_in, str):
				names_in = (names_in,)

			if make_unique:
				_args = [str(x) for x in args]
				_rsuffix = prepare_names(rsuffix, [], make_unique=False) if rsuffix is not None else []
				return ["_".join([n] + _args + _rsuffix) for n in names_in]
			else:
				return list(names_in)

		@wraps(func)
		def wrapper(*args, column_names=default_column_names, make_unique_name=default_make_unique,
					rsuffix=default_rsuffix, no_join=False, **kwargs):
			_ret_series = func(*args, **kwargs)
			_col_names = column_names

			# Try with column_names, if not present use col_names from decorator, or the dataframe columns
			_col_names = prepare_names(_col_names, args[1:], make_unique=make_unique_name, rsuffix=rsuffix)
			# logger.debug(f"Col names        {str(list(_ret_series.columns))} -> {str(_col_names)}")
			_ret_series.columns = _col_names
			args[0].dataframe.drop(columns=_col_names, errors='ignore', axis=1, inplace=True)
			if not no_join:
				args[0].dataframe = args[0].dataframe.join(_ret_series)
			return _ret_series

		return wrapper

	return decorator


def save_to_group(default_group_name):
	# type: (str) -> callable
	""" Saves the dataframe column names to list of similar attributes. It also creates a getter function for the list if
	it is not present yet. """

	def decorator(func):
		@wraps(func)
		def wrapper(*args, group_name=default_group_name, label=None, **kwargs):
			_ret_dataframe = func(*args, **kwargs)
			_group_name = group_name
			if _group_name is None:
				# User specified, it is not allowed to add it to the group
				return _ret_dataframe
			col_names = list(_ret_dataframe.columns.to_list())
			params_called_with = list(args[1:])
			if label is not None:
				if type(label) in [list, tuple]:
					params_called_with += label
				else:
					params_called_with += [label]
			args[0].groups[_group_name].append((col_names, tuple(params_called_with)))
			return _ret_dataframe

		return wrapper

	return decorator
