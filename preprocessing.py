import os
from typing import Union, Tuple, Optional, List

from numpy import single

from stocker import Stock
import numpy as np
import pickle
import datetime as dt
import random
import logging

logger = logging.getLogger(__name__)


def read_or_raise(paths):
	# type: (List[str]) -> Optional[List[np.array]]
	if all(os.path.exists(x) for x in paths):
		logger.info("Loading preprocessed data")
		r = []
		for name in paths:
			with open(name, 'rb') as rf:
				r.append(pickle.load(rf))
		return r
	raise FileNotFoundError


def save_pickle(paths, objs):
	for name, obj in zip(paths, objs):
		with open(name, 'wb') as wf:
			pickle.dump(obj, wf)


def split_tuples(data, dtype=None, dtype2=None):
	# type: (Union[np.array, list], Optional[np.dtype, single], Optional[np.dtype, single]) -> Tuple[np.array, np.array]
	"""
	:param data: [(X, y), ...]
	:param dtype: datatype for X
	:param dtype2: datatype for y
	:return: Tuple of np arrays splited on axis 1
	"""
	new_X = []
	new_y = []
	for x, y in data:
		new_X.append(x)
		new_y.append(y)
	return np.array(new_X, dtype=dtype), np.array(new_y, dtype=dtype2 if dtype2 else dtype)


def equalize_sets(data, func):
	# type: (Union[np.array, list], callable) -> Union[np.array, list]
	"""  equalize targets  """
	first = []
	second = []

	for x, y in data:
		if func(y):
			first.append((x, y))
		else:
			second.append((x, y))

	min_pivot = min(len(first), len(second))
	print(f"{len(first)} of first type, {len(second)} of second type.")

	random.shuffle(first)
	random.shuffle(second)
	first, second = first[:min_pivot], second[:min_pivot]
	ret = first[:min_pivot]+second[:min_pivot]
	random.shuffle(ret)
	return ret


def create_preprocessed_df_binary(stock, seq_len):
	# type: (Stock, int) -> Tuple[np.array, np.array]
	# Expected : ["log_mid_delta_preceding","RIC","target"]

	work_array = stock.prediction_model_data().values
	# work_array = np.expand_dims(work_array, axis=2)
	os.makedirs("preprocessed/", exist_ok=True)
	csv_name = f"preprocessed/{stock.ticker}-{dt.date.today()}"
	paths = [csv_name + "-X.binary.nparray.pickle", csv_name + "-y.binary.nparray.pickle"]
	try:
		return read_or_raise(paths)
	except FileNotFoundError:
		pass  # create the files
	data = []
	for i in range(seq_len, work_array.shape[0]):
		log_deltas = work_array[i - seq_len:i, 0].astype(np.float32)
		target = work_array[i, 2, np.newaxis]
		data.append((log_deltas, target))

	data = equalize_sets(data, lambda target: target[0] > 0)

	"""  shuffle rows  """
	random.shuffle(data)
	X_ret, y_ret = split_tuples(data, np.float32)
	X_ret = np.expand_dims(X_ret, axis=2)
	y_ret = np.expand_dims(y_ret, axis=2)
	save_pickle(paths, [X_ret, y_ret])
	return X_ret, y_ret
