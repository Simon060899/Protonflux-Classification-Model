import pandas as pd
import numpy as np
from numpy import sin, cos, pi

R_EARTH = 6371  # km

def preprocess_customer_data(df):
    """
    Initial preprocessing for customer data:
    - Normalizes coordinates if they are in km.
    - Calculates radial distance 'rdist'.
    - Computes '|z|' (absolute value of z-coordinate).
    """
    df = df.copy()
    # Check for km coordinates and normalize
    if 'x(km)' in df.columns and 'y(km)' in df.columns and 'z(km)' in df.columns:
        print("Normalizing coordinates from km to Earth radii.")
        for coord in ['x', 'y', 'z']:
            df[coord] = df[f'{coord}(km)'] / R_EARTH
        # Drop original km columns if you want to, or keep them
        # df.drop(columns=[f'{c}(km)' for c in ['x', 'y', 'z']], inplace=True)
    elif not all(c in df.columns for c in ['x', 'y', 'z']):
        raise ValueError("DataFrame must contain 'x', 'y', 'z' coordinates or 'x(km)', 'y(km)', 'z(km)' coordinates.")

    if 'rdist' not in df.columns:
        print("Calculating 'rdist'.")
        df['rdist'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

    if '|z|' not in df.columns:
        print("Calculating '|z|'.")
        df = compute_magnitude(df, 'z')
            
    return df

def compute_magnitude(df, column_name):
    """
    Compute absolute value of a specified column.
    """
    df = df.copy()
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame for magnitude calculation.")
    df[f'|{column_name}|'] = np.abs(df[column_name])
    print(f"Computed magnitude (absolute value) of {column_name}")
    return df

def calculate_longitude_and_latitude(df):
    """
    Calculate latitude and longitude from Cartesian coordinates (x, y, z).
    Assumes x, y, z are in Earth radii or consistent units for angle calculation.
    This is not the same as geographic coordinates, because we are using the GSE coordinate system.
    """
    df = df.copy()
    required_cols = ['x', 'y', 'z']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for geographic coordinates: {missing_cols}")

    r = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    # Avoid division by zero or invalid values for arcsin/arccos if r is zero
    # For r=0, angles are undefined; here we might set to 0 or NaN.
    # Using np.divide to handle r=0, resulting in NaN, which is often appropriate.
    
    # Latitude: angle from equator (-90° to 90°)
    df['latitude'] = np.degrees(np.arcsin(np.clip(np.divide(df['z'], r, out=np.zeros_like(df['z'], dtype=float), where=r!=0), -1.0, 1.0)))
    
    # Longitude: angle from prime meridian (-180° to 180°)
    df['longitude'] = np.degrees(np.arctan2(df['y'], df['x']))
    
    print("Calculated coordinates (latitude, longitude) in GSE coordinate system")
    return df

def compute_P107(df):
    """
    Compute smoothed F10.7 solar flux index (P10.7) using 81-day centered moving average.
    Requires 'F107' column and a DatetimeIndex.
    """
    df = df.copy()
    if 'F107' not in df.columns:
        raise ValueError("Column 'F107' not found, cannot compute 'P107'.")
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: DataFrame does not have a DatetimeIndex. P107 results may be incorrect or fail.")
        # Attempt to convert if it's a common datetime column name
        if 'datetime[UTC]' in df.columns and df.index.name != 'datetime[UTC]':
             try:
                 df.index = pd.to_datetime(df['datetime[UTC]'])
                 print("Converted 'datetime[UTC]' column to DatetimeIndex.")
             except Exception as e:
                 raise ValueError(f"Failed to convert 'datetime[UTC]' to DatetimeIndex for P107 calculation: {e}")
        elif df.index.name is None and len(df.index) > 0 : # if index is just RangeIndex
             raise ValueError("DataFrame index is not a DatetimeIndex. P107 calculation requires a DatetimeIndex.")


    # Calculate centered 81-day moving average (40 days before, current day, 40 days after)
    # Ensure the rolling window works even with sparse data by using min_periods=1
    centered_avg = df['F107'].rolling('81D', center=True, min_periods=1).mean()
    
    df['P107'] = (df['F107'] + centered_avg) / 2
    print("Computed smoothed F10.7 solar flux index (P107)")
    return df

def calculate_specific_polynomial_features_from_list(df, features_list=[]):
    """
    Calculate polynomial features using NumPy's optimized operations based on a list of desired features.
    Input format example for features_list:
    ['x', 'VxSW_GSE', 'Pdyn', 'FootType', 'rdist', 'x*y', 'x*VxSW_GSE', 'y^2', etc.]
    """
    df = df.copy()
    base_features_required = set()
    poly_feature_definitions = [] # Stores tuples of (output_name, type, inputs)
                                  # e.g. ('x*y', 'mul', ['x','y']), ('y^2', 'sq', ['y'])

    print("\\nProcessing feature list for polynomial generation...")
    for feature_str in features_list:
        if '*' in feature_str: # Interaction term
            parts = feature_str.split('*')
            if len(parts) == 2:
                base_features_required.add(parts[0])
                base_features_required.add(parts[1])
                poly_feature_definitions.append({'name': feature_str, 'type': 'interaction', 'inputs': parts})
            else:
                print(f"Warning: Cannot parse interaction feature '{feature_str}'. Skipping.")
        elif '^' in feature_str: # Polynomial term (e.g., x^2)
            parts = feature_str.split('^')
            if len(parts) == 2:
                try:
                    degree = int(parts[1])
                    if degree == 2: 
                        base_features_required.add(parts[0])
                        poly_feature_definitions.append({'name': feature_str, 'type': 'squared', 'inputs': [parts[0]]})
                    else:
                        print(f"Warning: Polynomial feature '{feature_str}' with degree {degree} not automatically supported. Skipping.")
                except ValueError:
                    print(f"Warning: Cannot parse degree for polynomial feature '{feature_str}'. Skipping.")
            else:
                print(f"Warning: Cannot parse polynomial feature '{feature_str}'. Skipping.")
        else: # Base feature, ensure it's noted if not already
            base_features_required.add(feature_str)

    # Check if all base features needed for polynomials exist in df
    missing_base_for_poly = [f for f in base_features_required if f not in df.columns]
    if missing_base_for_poly:
        # Check if these missing base features are actually part of the original features_list.
        # If so, it's a problem. If not, they were only inferred for poly terms.
        truly_missing_and_requested = [f for f in missing_base_for_poly if f in features_list]
        if truly_missing_and_requested:
            raise ValueError(f"Missing required base features in DataFrame for polynomial generation and direct use: {truly_missing_and_requested}")
        else:
            # These base features are only needed for polynomials but are missing.
            # This means some requested poly terms cannot be built.
             print(f"Warning: Base features required for some polynomial terms are missing: {missing_base_for_poly}. Some polynomial features may not be generated.")


    newly_created_feature_names = []
    for poly_def in poly_feature_definitions:
        # Only create if not already present and inputs are available
        if poly_def['name'] not in df.columns and all(inp in df.columns for inp in poly_def['inputs']):
            if poly_def['type'] == 'interaction':
                f1, f2 = poly_def['inputs']
                df[poly_def['name']] = df[f1] * df[f2]
                newly_created_feature_names.append(poly_def['name'])
            elif poly_def['type'] == 'squared':
                base = poly_def['inputs'][0]
                df[poly_def['name']] = df[base] ** 2
                newly_created_feature_names.append(poly_def['name'])
    
    if newly_created_feature_names:
        print(f"Calculated and added {len(newly_created_feature_names)} polynomial/interaction features: {newly_created_feature_names}")
    else:
        print("No new polynomial/interaction features were added (either already exist or base features missing).")
        
    return df, newly_created_feature_names


def create_mixed_model_predictions(df):
    """
    Create mixed model predictions by averaging XGBoost and Logistic Regression probabilities
    and generating binary classifications based on a 0.5 threshold.
    Expects columns like 'prob_xgb_10', 'prob_logistic_10', etc.
    """
    df = df.copy()
    expected_prob_cols = {
        10: ['prob_xgb_10', 'prob_logistic_10'],
        100: ['prob_xgb_100', 'prob_logistic_100']
    }
    all_cols_created_successfully = True

    for threshold, prob_cols in expected_prob_cols.items():
        if all(col in df.columns for col in prob_cols):
            df[f'prob_mixed_{threshold}'] = (df[prob_cols[0]] + df[prob_cols[1]]) / 2
            df[f'high_risk_mixed_{threshold}'] = (df[f'prob_mixed_{threshold}'] >= 0.5).astype(int)
            print(f"✓ Successfully created mixed model predictions for threshold {threshold}")
            
            # Verify probabilities are within [0,1]
            if df[f'prob_mixed_{threshold}'].min() < 0 or df[f'prob_mixed_{threshold}'].max() > 1:
                print(f"⚠ Warning: Column prob_mixed_{threshold} contains values outside the valid probability range [0,1]")
                all_cols_created_successfully = False
        else:
            missing = [col for col in prob_cols if col not in df.columns]
            print(f"⚠ Warning: Missing probability columns to create mixed model for threshold {threshold}: {missing}")
            all_cols_created_successfully = False
            # Create placeholder columns if inputs are missing to avoid downstream errors if these columns are expected
            if f'prob_mixed_{threshold}' not in df.columns: df[f'prob_mixed_{threshold}'] = np.nan
            if f'high_risk_mixed_{threshold}' not in df.columns: df[f'high_risk_mixed_{threshold}'] = np.nan


    if all_cols_created_successfully:
        print("✓ All mixed model columns processed (or placeholders created if inputs were missing).")
    else:
        print("⚠ Some mixed model predictions could not be fully generated due to missing input probability columns.")
        
    return df

