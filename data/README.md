# Data Repository
This directory contains the time-series datasets used to train the demand prediction model.

## Dataset Sources
* `Charging station_A_Calif.csv`: Hourly usage logs (US Date Format).
* `Charging station_B__Calif.csv`: Hourly usage logs (ISO Date Format).
* `Charging station_C__Calif.csv`: Hourly usage logs (ISO Date Format).

## Data Dictionary
| Feature | Description |
| :--- | :--- |
| `EV Charging Demand (kW)` | Target variable; total power requested per hour. |
| `Grid Stability Index` | Measure of local grid reliability. |
| `Electricity Price` | Market cost per kWh at the time of charging. |
| `Weather Conditions` | Environmental factors affecting battery efficiency. |