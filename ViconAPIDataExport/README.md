# Usage

## Setup python venv
In the [Python Prototype Folder](https://github.com/MobMonRob/MarkerTrackingGAStudien/tree/main/python-prototype-LA) run 
```
$ python -m venv .venv
```
Then run the venv activation script 
```
$ source .venv/bin/activate
```

## Setup reqs
```
(.venv)$ pip install -r requirements.txt
```

## Run the playground
The playground script can only work with **2** camera angles on **one** single Marker at the moment, which corresponds to the xcp/csv files as shown in the example command.
```
(.venv)$ python playground.py -xcp "camera-configs/experiment-1.xcp"
-centroids "csvdata/experiment-1/centroid-dump-1-marker-cam-4-and-7.csv"
-markers "csvdata/experiment-1/marker-dump-1-marker-cam-4-and-7.csv"
```
The visualisation in the playground script can accomodate for more than 2 cameras and multiple markers, but the lvp computation does not account for that. 
