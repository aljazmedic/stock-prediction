## Predvidevanje vrednosti delnic s tensorflowom v pythonu

Za zagon na windowsu je potrebno namestiti
[python 3.6](https://www.python.org/downloads/release/python-360/")

Ko je Python nameščen, je potrebno posodobiti nameščevalec knjižnic pip
```cmd
python -m pip install --upgrade pip
```
Nato je potrebno namestiti vse potrebne knjižnice<br>
V ukazni vrstici vstopimo v direktorij projekta in zaženemo
```cmd
pip install -r requirements.txt
```
Program za pravilno delovanje potrebuje tudi API ključe. Vsakega izmed njih lahko dobimo z registracijo na spletne strani, napisane v .env.preview.
Ključe vnesemo med narekovaje v datoteko in datoteko preimenujemo
```cmd
move .env.preview .env
``` 

Ko so ključi na mestu in se knjižnice namestijo, lahko z zagonom
`stock` in parametri zaženemo program

### Uporaba ukaza stock
`stock -h` prikaže vse možne ukaze<br>
Ima dve glavni funkcionalnosti:
##### Prikaz grafov delnic z indikatorji

  - Prikaz delnice AMZN<br>
	`stock display --stocks AMZN`

  - Prikaz delnice MSFT <br>z indikatorjem SMA z 10-dnevnim oknom:<br>
	`stock display --stocks MSFT --indicators SMA.10`
	
  - Prikaz delnice AAPL <br>z indeikatorjem VWAP z 19-dnevnim oknom <br>z eksplicitnim izpisom programa<br>
	`stock -v display --stocks AAPL --indicators VWAP.19`
##### Predvidevanje delnic

  - Kreacija modela na podlagi delnice AMZN<br>
	`stock predict AMZN`

  - Nalaganje modela na podlagi delnice AAPL, prikazom na grafu<br>
	`stock predict AAPL --display`
	
  - Treniranje modela na podlagi delnice MSFT, prikazom na grafu <br>
	`stock predict MSFT --display --train`