# Avrami fit

Package designed to calculate the crystallization kinetic rate of semi-crystalline polymer blends

## Installation
```
pip install avrami 
```

## Example usage
```
from avrami.main import avrami_data_process

file_path = your .csv file

initial_guess = enthalpy, k, tzero, n

result = avrami_data_process(file_path, initial_guess)
```

