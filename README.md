## Stock prediction in python

### Setup
In order to run the stock prediction,
[python 3.6.8](https://www.python.org/downloads/release/python-368/)
has to be installed

All the dependencies are found in requirements.txt file, install with
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```
You need to set up api keys for Alpaca and Alphavantage.
Sample `.env.preview` file can be found in root directory.
Fill in your values and rename it:
```bash
# MS-DOS
move .env.preview .env
# bash
mv .env.preview .env
``` 

You can run the problem using `stock` and giving it the paramteres

### Usage
`stock -h` shows all possible flags and commands<br>
#### Main two functions are:
##### Display of graphs for given stock

  - Show graf of the stock AMZN<br>
	`stock display --stocks AMZN`

  - Show graph of the stock MSFT <br>
  with indicator SMA (10-day window)<br>
	`stock display --stocks MSFT --indicators SMA.10`
	
  - Show graf of the stock AAPL
  <br>with indicator VWAP (19-day window)
  <br>and verbose output<br>
	`stock -v display --stocks AAPL --indicators VWAP.19`
##### Stock prediction

  - Create a model based on the stock AMZN<br>
	`stock predict AMZN`

  - Create a model based on the stock AAPL,<br>display it<br>
	`stock predict AAPL --display`
	
  - Create and train a model based on the stock MSFT,<br>
  display it <br>
	`stock predict MSFT --display --train`

