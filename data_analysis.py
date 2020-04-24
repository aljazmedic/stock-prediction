# %matplotlib inline
import logging
import os
import time
from typing import Tuple, List, Optional

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from preprocessing import create_preprocessed_df_binary
from stock import Stock
from stock_server import StockServer

logger = logging.getLogger(__name__)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow.keras').setLevel(logging.CRITICAL)


def test_model_on_stock(stock, model, seq_len, slc=slice(-10, None)):
	# type: (Stock, Sequential, int, Optional[slice]) -> Tuple[Tuple[np.array,np.array],Tuple[np.array, np.array],Tuple[np.array,np.array,np.array,np.array]]
	work_array = stock.prediction_model_data(dropna=False).values

	pred_range = slc.indices(work_array.shape[0]-seq_len)
	pred_range = range(seq_len + pred_range[0], seq_len + pred_range[1], pred_range[2])

	logger.debug(f'Dataset shape: {work_array.shape}')
	logger.debug(f'Prediction slice: {slc}')
	logger.debug(f'Prediction range: {pred_range}')
	# model input, output
	_X_in = np.array([
		work_array[i - seq_len:i, 0].astype(np.float32)
		for i in pred_range
	])  # Data to be inputted to model
	_X_in = np.expand_dims(_X_in, axis=2)
	_y_out = model.predict(_X_in)
	_y_predicted = [
		int(tf.math.argmax(y_pred))
		for y_pred in _y_out
	]

	_closes_in = [
		work_array[i - seq_len:i, 3].astype(np.float32)
		for i in pred_range
	]  # close prices corresponding to logs of changes
	_prediction_times = work_array[slc, 4]
	_close_on_prediction = work_array[slc, 3]  # close prices of predictions
	_y_actual = work_array[slc, 2]
	# all stock data
	_all_times = work_array[:, 4]
	_all_closes = work_array[:, 3]
	return (_X_in, _y_out), (_all_times, _all_closes), (_prediction_times, _close_on_prediction, _y_predicted, _y_actual)


def create_model(seq_len):
	"""   Model creation   """
	m = Sequential([
		LSTM(128, input_shape=(seq_len, 1), return_sequences=True),
		Dropout(0.2),
		BatchNormalization(),

		LSTM(128, input_shape=(seq_len, 1), return_sequences=True),
		Dropout(0.1),
		BatchNormalization(),

		LSTM(128, input_shape=(seq_len, 1)),
		Dropout(0.2),
		BatchNormalization(),

		Dense(32, activation='relu'),
		Dropout(0.2),

		Dense(2, activation='softmax')
	])

	opt = Adam(decay=1e-6)
	m.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])

	return m


def get_callbacks(name, verbose=10):
	# type: (str, Optional[int]) -> List[Callback, ...]
	os.makedirs("models/", exist_ok=True)
	tensorboard = TensorBoard(log_dir=os.path.join('logs', name))
	checkpoint = ModelCheckpoint(
		filepath=os.path.join("models", "checkpoint", f"RNN_{name}-" + "{val_acc:.3f}.ckpt"),
		verbose=verbose, save_best_only=True, monitor='val_acc',
		save_weights_only=True, mode='max')  # saves only the best ones
	return [tensorboard, checkpoint]


def main(stock, seq_len, epochs, batch_size, action='train', show=False, verbose=10):
	NAME = f'{stock.ticker}-{seq_len}-SEQ-{int(time.time())}'

	X_data, y_data = create_preprocessed_df_binary(stock, seq_len)
	X_train, X_test, y_train, y_test = train_test_split(
		X_data,
		y_data,
		test_size=0.3)

	X_train, X_val, y_train, y_val = train_test_split(
		X_train,
		y_train,
		test_size=0.16)

	model = create_model(seq_len)
	loss, acc = model.evaluate(X_test, y_test, verbose=verbose)
	model_stage = "Untrained"
	logger.info("{} model, loss: {:5.2f}%, accuracy: {:5.2f}%".format(model_stage, 100 * loss, 100 * acc))

	# Display the model's architecture
	# plot_model(model, to_file='./model.png', rankdir='LR', dpi=300) # requires install graphviz and pydot
	# model.summary()

	if action == 'load':
		# Loading
		latest = tf.train.latest_checkpoint(os.path.join('models', 'checkpoint'))
		logger.info(f"Latest checkpoint file: {latest}")
		if latest is not None:
			model.load_weights(latest)
			loss, acc = model.evaluate(X_test, y_test, verbose=verbose)
			model_stage = "Loaded"
			logger.info("{} model, loss: {:5.2f}%, accuracy: {:5.2f}%".format(model_stage, 100 * loss, 100 * acc))
		else:
			# no checkpoint data
			action = 'train'

	if action == 'train':
		model.fit(
			X_train, y_train,
			batch_size=batch_size,
			epochs=epochs,
			validation_data=(X_val, y_val),
			callbacks=get_callbacks(NAME, verbose=verbose),
			verbose=verbose
		)
		loss, acc = model.evaluate(X_test, y_test, verbose=verbose)
		model_stage = "Trained"
		logger.info("{} model, loss: {:5.2f}%, accuracy: {:5.2f}%".format(model_stage, 100 * loss, 100 * acc))
		model.save(filepath=f'models/{NAME}.h5')

	prediction_slice = slice(-1, None)
	if show:
		prediction_slice = slice(-40, None)
	model_io_data, _, predictions = test_model_on_stock(stock, model, seq_len, slc=prediction_slice)

	print("{:5.2f}% verjetnost padanja, {:5.2f}% verjetnost naraščanja".format(*model_io_data[1][-1]*100))
	if show:
		from stock_display import graph_stock
		import matplotlib.pyplot as plt
		from pandas.plotting import register_matplotlib_converters
		register_matplotlib_converters()

		graph_stock(stock, predictions, title='{} ({:5.2f}% acc)'.format(stock.ticker.upper(), acc*100))
		plt.show()


if __name__ == '__main__':
	world_stock_server = StockServer()

	stock = world_stock_server["AAPL"]

	seq_len = 48
	epochs = 50
	batch_size = 64
	main(stock, seq_len, epochs, batch_size, show=True, action='train')
