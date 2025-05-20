import pandas as pd
import numpy as np
import joblib
import pathlib
import helper_functions # Assumes helper_functions.py is in the same directory
from sklearn.preprocessing import StandardScaler # For scaler reconstruction if needed

# Define base path for the package structure
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

# Define model filenames and their types/thresholds
MODEL_CONFIG = {
    'xgb_10': {
        'file': '10_xgboost_model_23_feat+P107+long+lat.joblib',
        'type': 'xgb',
        'threshold': 10
    },
    'xgb_100': {
        'file': '100_xgboost_model_11_feat+P107+long+lat.joblib',
        'type': 'xgb',
        'threshold': 100
    },
    'logistic_10': {
        'file': '10_logistic_regression_model_26_feat_base+P107.joblib',
        'type': 'logistic',
        'threshold': 10
    },
    'logistic_100': {
        'file': '100_logistic_regression_model_16_feat_base+P107+abs(long)+abs(lat).joblib',
        'type': 'logistic',
        'threshold': 100
    }
}

def load_models():
    """Load all models specified in MODEL_CONFIG."""
    loaded_models = {}
    print("Loading models...")
    for model_key, config in MODEL_CONFIG.items():
        model_path = MODELS_DIR / config['file']
        try:
            model_data = joblib.load(model_path)
            loaded_models[model_key] = {
                'model': model_data['model'],
                'features': model_data['features'],
                'scaler': model_data.get('scaler'), 
                'type': config['type'],
                'threshold': config['threshold']
            }
            print(f"Successfully loaded model: {config['file']}")
            print(f"  Type: {config['type']}, Threshold: {config['threshold']}")
            print(f"  Features ({len(model_data['features'])}): {model_data['features'][:3]}...")
        except FileNotFoundError:
            print(f"ERROR: Model file not found: {model_path}")
            print("Please ensure all model files are in the 'models' directory.")
            raise
        except Exception as e:
            print(f"ERROR: Could not load model {config['file']}: {e}")
            raise
    if not loaded_models:
        print("No models were loaded. Exiting.")
        exit()
    return loaded_models

def main():
    """Main script to load data, preprocess, predict, and generate mixed model outputs."""
    # Load models
    models_data = load_models()

    # Load sample customer data
    sample_data_path = DATA_DIR / 'sample_data.csv'
    print(f"\\nLoading sample data from: {sample_data_path}")
    try:
        df = pd.read_csv(
            sample_data_path, 
            sep=',', 
            parse_dates=['datetime[UTC]'], 
            index_col='datetime[UTC]'
        )
        print("Sample data loaded successfully with DatetimeIndex.")
        
        # Explicitly convert expected numeric columns to numeric, coercing errors to NaN for now
        # This helps identify if any non-numeric data slipped through for some reason.
        # The original headers are known from the sample data creation.
        # Model features will later determine which of these are strictly needed.
        expected_numeric_cols = ['x(km)', 'y(km)', 'z(km)', 'AE_index', 'Pdyn', 'Dst_index', 'VxSW_GSE', 'F107', 'FootType']
        for col in expected_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce will turn bad data into NaNs
            else:
                print(f"Warning: Expected column '{col}' not found in loaded data for numeric conversion.")
        
        # Check for NaNs introduced by coercion in columns vital for initial steps
        if df[[ 'x(km)', 'y(km)', 'z(km)', 'F107']].isnull().any().any():
            print("WARNING: NaNs detected in critical coordinate or F107 columns after loading and numeric conversion. This may cause errors downstream.")
            print(df[[ 'x(km)', 'y(km)', 'z(km)', 'F107']].isnull().sum())

    except FileNotFoundError:
        print(f"ERROR: Sample data file not found: {sample_data_path}")
        print("Please ensure 'sample_customer_data.csv' is in the 'data' directory.")
        return
    except Exception as e:
        print(f"ERROR: Could not load sample data: {e}")
        return

    print(f"Initial sample data shape: {df.shape}")
    print(f"Initial columns: {df.columns.tolist()}")

    # === Start Preprocessing ===
    print("\\n--- Starting Data Preprocessing ---")

    # 1. Initial preprocessing (coordinates, rdist, |z|)
    try:
        df = helper_functions.preprocess_customer_data(df)
    except ValueError as e:
        print(f"Error during initial preprocessing: {e}")
        return

    # 2. Calculate coordinates
    try:
        df = helper_functions.calculate_longitude_and_latitude(df)
    except ValueError as e:
        print(f"Error calculating longitude and latitude: {e}. Ensure x,y,z are present.")
        return
        
    # 3. Calculate absolute latitude/longitude
    if 'latitude' in df.columns:
        df['|latitude|'] = df['latitude'].abs()
        print("Calculated '|latitude|'.")
    else:
        print("Warning: 'latitude' column not found for calculating '|latitude|'.")

    if 'longitude' in df.columns:
        df['|longitude|'] = df['longitude'].abs()
        print("Calculated '|longitude|'.")
    else:
        print("Warning: 'longitude' column not found for calculating '|longitude|'.")

    # 4. Calculate P107
    if 'P107' not in df.columns:
        if 'F107' in df.columns:
            print("Column 'P107' not found, attempting to calculate from 'F107'.")
            try:
                df = helper_functions.compute_P107(df)
            except ValueError as e:
                print(f"Error computing P107: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during P107 computation: {e}")
        else:
            print("Warning: 'P107' and 'F107' not found. P107 cannot be calculated.")
    else:
        print("Column 'P107' already present in the data.")

    # 5. Filter by radial distance (rdist >= 6 RE)
    if 'rdist' in df.columns:
        initial_row_count = len(df)
        df = df[df['rdist'] >= 6].copy() # Use .copy() to avoid SettingWithCopyWarning
        removed_row_count = initial_row_count - len(df)
        percent_removed = (removed_row_count / initial_row_count * 100) if initial_row_count > 0 else 0
        print(f"\\nFiltering data for rdist >= 6 RE:")
        print(f"  Removed {removed_row_count} rows ({percent_removed:.2f}%) with rdist < 6 RE.")
        print(f"  Remaining rows: {len(df)}")
        if len(df) == 0:
            print("WARNING: No data remaining after rdist filtering. Predictions cannot proceed.")
            
    else:
        print("Warning: 'rdist' column not found. Cannot apply rdist >= 6 RE filter.")

    print("--- Data Preprocessing Complete ---")        
    print(f"Data shape after preprocessing & filtering: {df.shape}")
    print(f"Columns after preprocessing: {df.columns.tolist()}")

    all_generated_poly_interaction_features = set()

    # === Apply Models and Generate Predictions ===
    print("\\n--- Applying Models and Generating Predictions ---")

    for model_key, model_info in models_data.items():
        print(f"\\nProcessing model: {model_key} (File: {MODEL_CONFIG[model_key]['file']})")
        required_features = model_info['features']
        
        # Check for missing features and generate if polynomial/interaction
        current_cols_in_df = df.columns.tolist()
        missing_features = [feat for feat in required_features if feat not in current_cols_in_df]

        if missing_features:
            print(f"Missing features for model {model_key}: {missing_features}")
            print("Attempting to generate polynomial/interaction features...")
            try:
                df, newly_added = helper_functions.calculate_specific_polynomial_features_from_list(df, required_features)
                if newly_added: # newly_added is a list of strings (feature names)
                    all_generated_poly_interaction_features.update(newly_added)
                # Re-check missing features
                current_cols_in_df = df.columns.tolist()
                missing_features = [feat for feat in required_features if feat not in current_cols_in_df]
                if missing_features:
                    print(f"ERROR: Still missing base features after attempting generation for model {model_key}: {missing_features}")
                    print(f"Skipping predictions for model {model_key}.")
                    continue # Skip to next model
            except ValueError as e:
                print(f"ERROR: Could not generate features for model {model_key}: {e}")
                print(f"Skipping predictions for model {model_key}.")
                continue
            except Exception as e:
                print(f"ERROR: An unexpected error occurred during feature generation for model {model_key}: {e}")
                print(f"Skipping predictions for model {model_key}.")
                continue
        
        try:
            X = df[required_features]
            prob_col_name = f"prob_{model_info['type']}_{model_info['threshold']}"
            
            if model_info['type'] == 'logistic':
                scaler = model_info['scaler']
                if scaler is None:
                    print(f"ERROR: Scaler not found for logistic model {model_key}. Cannot make predictions.")
                    continue
                
                
                categorical_features = [col for col in required_features if col == 'FootType'] 
                numerical_features = [col for col in required_features if col not in categorical_features]
                
                X_numerical = X[numerical_features].values
                X_numerical_scaled = scaler.transform(X_numerical) # Scaler expects numpy array
                
                X_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=X.index)
                
                # Add categorical features back if they exist
                for cat_col in categorical_features:
                    if cat_col in X.columns:
                         X_scaled_df[cat_col] = X[cat_col].values # Ensure correct alignment
                
                # Ensure final X_scaled has columns in the order of required_features
                X_to_predict = X_scaled_df[required_features].values

                df[prob_col_name] = model_info['model'].predict_proba(X_to_predict)[:, 1]
            
            elif model_info['type'] == 'xgb':
                # XGBoost models were trained with enable_categorical=True if FootType was used.
                df[prob_col_name] = model_info['model'].predict_proba(X.values)[:, 1]
            
            print(f"Generated probabilities column: {prob_col_name}")

        except Exception as e:
            print(f"ERROR: Failed to make predictions for model {model_key}: {e}")
            print(f"  Required features: {required_features}")
            print(f"  Features in X: {X.columns.tolist() if isinstance(X, pd.DataFrame) else 'Numpy array'}")
            import traceback
            traceback.print_exc()
            continue

    # === Create Mixed Model Predictions ===
    print("\\n--- Creating Mixed Model Predictions ---")
    df = helper_functions.create_mixed_model_predictions(df)

    # === Display Results ===
    print("\\n--- Final DataFrame with Predictions (First 5 rows) ---")
    
    # Select columns to display: original + all generated probability and high_risk columns
    output_columns = [col for col in df.columns if 'prob_' in col or 'high_risk_' in col]
    
    original_keys_to_show = ['x', 'y', 'z', 'F107', 'P107', 'latitude', 'longitude', 'rdist', '|z|']
    display_cols = [col for col in original_keys_to_show if col in df.columns] + output_columns
    
    # Ensure no duplicate columns
    display_cols = sorted(list(set(display_cols)), key=lambda x: (x not in output_columns, x))

    print(df[display_cols].head())
    
    print("\\n--- Prediction Statistics (Positive Cases for high_risk columns) ---")
    for col in df.columns:
        if 'high_risk_' in col:
            if df[col].dtype == 'int' or df[col].dtype == 'bool': # Check if it's a binary classification column
                positive_count = df[col].sum()
                positive_percent = (positive_count / len(df)) * 100 if len(df) > 0 else 0
                print(f"{col}: {positive_count} positive predictions ({positive_percent:.2f}%)")
            
    # Save the DataFrame with predictions, excluding generated polynomial/interaction terms
    output_prediction_file = DATA_DIR / 'sample_data_with_predictions.csv'
    try:
        features_to_drop_before_saving = [feat for feat in all_generated_poly_interaction_features if feat in df.columns]
        if features_to_drop_before_saving:
            print(f"\\nDropping the following generated polynomial/interaction features before saving: {features_to_drop_before_saving}")
            df_to_save = df.drop(columns=features_to_drop_before_saving)
        else:
            df_to_save = df.copy() # Save a copy if nothing to drop
        
        # Round prediction columns to 2 decimal places
        for col in df_to_save.columns:
            if col.startswith('prob_') or col.startswith('high_risk_'):
                # Ensure column is numeric before rounding to avoid errors with potential non-numeric placeholders
                if pd.api.types.is_numeric_dtype(df_to_save[col]):
                    df_to_save[col] = df_to_save[col].round(2)
                else:
                    print(f"Warning: Column '{col}' is not numeric and will not be rounded.")

        df_to_save.to_csv(output_prediction_file)
        print(f"\\nSuccessfully saved DataFrame with predictions to: {output_prediction_file}")
    except Exception as e:
        print(f"\\nERROR: Could not save DataFrame with predictions: {e}")

    print("\\nScript execution finished.")

if __name__ == '__main__':
    main() 