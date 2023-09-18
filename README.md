# RNN Simulator with NUMBA

## Introduction
This package provides an implementation of a recurrent neural network simulator with NUMBA.
The network can have multiple neural populations, different connectivity profiles (all to all, sparse, tuned, ...).
For more info look at the config files in ./conf

## Installation
Provide clear instructions on how to get your development environment running.
```bash
pip install -r requirements.txt
```
## Usage
Assuming the dependencies are installed, here is how to run the model (see notebooks folder or org folder for more doc)

```python
# import the network class
from src.model.rate_model import Network

# initialize model
model = Network(config_file_name, output_file_name, path_to_repo, **kwargs)

# kwargs can be any of the args in the config file

# run the model
model.run()
```


## Contributing
Feel free to contribute.
```
MIT License
Copyright (c) [2023] [A. Mahrach]
```
