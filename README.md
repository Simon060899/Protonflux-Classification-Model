## Author Information:

- **Author:** S. Mischel
- **Affiliation:** Ludwig-Maximilians-Universität München
- **Email:** S.Mischel@campus.lmu.de

# Customer Model Package: Soft Proton Flux Risk Classification

This package contains a model to predict soft proton flux risk at two different thresholds (10 and 100 cm^(-2)s^(-1)sr^(-1)keV^(-1).). It uses a mixed model approach, averaging predictions from XGBoost and Logistic Regression classifiers.

## Package Contents:

- `run_prediction.py`: The main Python script to load data and generate predictions.
- `helper_functions.py`: Contains utility functions for data preprocessing and feature engineering.
- `data/`: Directory containing sample input data.
  - `sample_data.csv`: A sample CSV file (100 data points) demonstrating the expected input format.
- `models/`: Directory where the pre-trained model files (`.joblib`) are placed.

## Prerequisites:

Make sure you have Python 3 installed along with the following libraries:
- pandas
- numpy
- scikit-learn (sklearn)
- joblib
- xgboost

You can typically install these using pip:
```bash
pip install pandas numpy scikit-learn joblib xgboost
```

## Input Data Format:

The script expects input data as a CSV file. The `sample_data.csv` in the `data/` directory serves as an example. 

Key required columns:
- `datetime[UTC]`: Timestamp in UTC (ISO 8601 format: `YYYY-MM-DDTHH:MM:SS`). This will be used as the index.
- `x(km)`, `y(km)`, `z(km)`: Spatial coordinates in kilometers in the GSE (Geocentric Solar Ecliptic) coordinate system. The script will convert these to Earth radii, as the models expect this unit as input.
- `F107`: F10.7 solar radio flux in solar flux units (sfu).
- `AE_index`: Auroral Electrojet index in nanotesla (nT).
- `Pdyn`: Solar wind dynamic pressure in nanopascals (nPa).
- `Dst_index`: Disturbance Storm Time index in nanotesla (nT).
- `VxSW_GSE`: Solar wind velocity X component in GSE in kilometers/second (km/s).
- `FootType`: Magnetic field line topology classification:
    - 0: Open field lines
    - 1: Solar wind field lines
    - 2: Closed field lines




The model predictions are for proton fluxes with energies in the range of **92.2 keV to 159.7 keV**.
The flux thresholds (10 and 100) are in units of cm^(-2)s^(-1)sr^(-1)keV^(-1).

The script will attempt to:
1. Normalize `x(km)`, `y(km)`, `z(km)` to Earth radii (creating `x`, `y`, `z`).
2. Calculate `rdist` (radial distance).
3. Calculate `|z|` (absolute value of the z-coordinate).
4. Calculate `latitude` and `longitude` from `x`, `y`, `z` (not geographic long and lat).
5. Calculate `|latitude|` and `|longitude|`.
6. Calculate `P107` from `F107` if `P107` is not already present.
7. **Important**: Filter the data to include only points where `rdist >= 6` Earth radii. A message will be printed indicating how many data points were removed by this filter.
8. Generate any missing polynomial or interaction features required by the specific models (e.g., `x*rdist`, `P107^2`).

## Model Files:

Find the following four `.joblib` model files in the `models/` directory:

1.  `10_xgboost_model_23_feat+P107+long+lat.joblib`
2.  `100_xgboost_model_11_feat+P107+long+lat.joblib`
3.  `10_logistic_regression_model_26_feat_base+P107.joblib`
4.  `100_logistic_regression_model_16_feat_base+P107+abs(long)+abs(lat).joblib`

These files contain the trained models and necessary scalers. The fist number indicates the choosen flux threshold for which the measurement is for.

## How to Run Predictions:

1.  Ensure all prerequisite libraries are installed.
2.  Place your input CSV data file in the `data/` directory (or modify the path in `run_prediction.py`). For testing, you can use the provided `sample_data.csv`.
3.  Make sure the four `.joblib` model files are in the `models/` directory.
4.  Navigate to the `Customer_Model_Package` directory in your terminal.
5.  Run the script:

    ```bash
    python run_prediction.py
    ```

## Output:

The script will print the following:
- Progress messages during model loading and data preprocessing.
- Messages about any features being generated (e.g., P107, polynomial features).
- The first 5 rows of the DataFrame containing the original key features and all generated probability and binary classification columns.
  - `prob_xgb_10`, `prob_logistic_10`: Individual model probabilities for threshold 10.
  - `prob_xgb_100`, `prob_logistic_100`: Individual model probabilities for threshold 100.
  - `prob_mixed_10`, `high_risk_mixed_10`: Mixed model probability and binary classification (>=0.5 threshold) for risk > 10 cm^(-2)s^(-1)sr^(-1)keV^(-1).
  - `prob_mixed_100`, `high_risk_mixed_100`: Mixed model probability and binary classification (>=0.5 threshold) for risk > 100 cm^(-2)s^(-1)sr^(-1)keV^(-1).
- Statistics on the number of positive predictions for each `high_risk` column.

This package aims to provide a straightforward way to apply the soft proton flux risk models to new data. 



