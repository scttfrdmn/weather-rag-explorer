#!/usr/bin/env python3

"""
Weather Data Cleaner

This module provides utilities for cleaning and validating weather data before
it's used in the RAG system, ensuring better quality results for user queries.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('weather_data_cleaner')

class WeatherDataCleaner:
    """
    Class for cleaning and validating weather data for RAG applications.
    
    This class provides methods to handle common issues in weather datasets:
    - Removing outliers
    - Handling missing values
    - Converting units
    - Normalizing data formats
    - Validating data ranges
    """
    
    # Define expected value ranges for different weather measurements
    VALID_RANGES = {
        'PRCP': (-0.1, 2000),  # Precipitation in mm (allowing small negative for rounding)
        'PRCP_MEAN': (-0.1, 2000),
        'PRCP_MAX': (0, 2000),
        'PRCP_MIN': (-0.1, 1000),
        'TMAX': (-90, 60),  # Temperature in Celsius
        'TMIN': (-90, 50),
        'LATITUDE': (-90, 90),
        'LONGITUDE': (-180, 180),
        'LATITUDE_MEAN': (-90, 90),
        'LONGITUDE_MEAN': (-180, 180)
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the weather data cleaner.
        
        Args:
            verbose (bool, optional): Whether to print detailed information during processing. 
                                     Defaults to False.
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate a weather dataframe.
        
        Args:
            df (pd.DataFrame): The input weather dataframe.
            
        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        if df is None or df.empty:
            logger.warning("Empty dataframe provided, nothing to clean")
            return df
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Log original data stats
        self._log_data_stats(cleaned_df, "before cleaning")
        
        # Perform cleaning steps
        cleaned_df = self._normalize_column_names(cleaned_df)
        cleaned_df = self._handle_dates(cleaned_df)
        cleaned_df = self._handle_missing_values(cleaned_df)
        cleaned_df = self._remove_outliers(cleaned_df)
        cleaned_df = self._validate_ranges(cleaned_df)
        
        # Log cleaned data stats
        self._log_data_stats(cleaned_df, "after cleaning")
        
        return cleaned_df
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to a standard format (uppercase).
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with normalized column names.
        """
        df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
        return df
    
    def _handle_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure dates are in a consistent datetime format.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with normalized dates.
        """
        if 'DATE' in df.columns:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
                    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                    
                # Remove rows with invalid dates
                invalid_dates = df['DATE'].isna()
                if invalid_dates.any():
                    n_invalid = invalid_dates.sum()
                    logger.warning(f"Removed {n_invalid} rows with invalid dates")
                    df = df[~invalid_dates]
            except Exception as e:
                logger.error(f"Error processing dates: {e}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values.
        """
        # Get initial count of NaN values
        initial_nan_count = df.isna().sum().sum()
        
        if initial_nan_count > 0:
            logger.info(f"Found {initial_nan_count} missing values")
            
            # For each column, handle missing values appropriately
            for col in df.columns:
                # Skip DATE column - we already handled it
                if col == 'DATE':
                    continue
                    
                # Count missing values in this column
                missing_count = df[col].isna().sum()
                if missing_count == 0:
                    continue
                
                # Different strategies based on column type and data
                if col in ['STATION', 'source_file']:
                    # For categorical columns, fill with 'Unknown'
                    df[col] = df[col].fillna('Unknown')
                elif col.startswith(('PRCP', 'TMAX', 'TMIN')):
                    # For numeric weather data, if < 25% missing, interpolate
                    if missing_count < len(df) * 0.25:
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    else:
                        # If many values missing, use 0 for precipitation, median for temp
                        if col.startswith('PRCP'):
                            df[col] = df[col].fillna(0)
                        else:
                            df[col] = df[col].fillna(df[col].median())
                elif col.startswith(('LATITUDE', 'LONGITUDE')):
                    # For geographic coordinates, interpolate if possible
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                else:
                    # For other numeric columns, use median
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        # For other string columns, use 'Unknown'
                        df[col] = df[col].fillna('Unknown')
        
        # Get final count of NaN values
        final_nan_count = df.isna().sum().sum()
        if final_nan_count > 0:
            logger.warning(f"Still have {final_nan_count} missing values after cleaning")
            
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from numerical columns.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        rows_before = len(df)
        
        # Only process numeric columns that represent weather data
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and 
                        any(col.startswith(prefix) for prefix in ['PRCP', 'TMAX', 'TMIN'])]
        
        for col in numeric_cols:
            if col in df.columns:
                # Use IQR method to identify outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds - using 3*IQR for more conservative removal
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Update extreme outliers to bounds instead of removing rows
                extreme_lower = df[col] < lower_bound
                extreme_upper = df[col] > upper_bound
                
                if extreme_lower.any() or extreme_upper.any():
                    num_outliers = extreme_lower.sum() + extreme_upper.sum()
                    
                    # Cap values rather than removing rows
                    df.loc[extreme_lower, col] = lower_bound
                    df.loc[extreme_upper, col] = upper_bound
                    
                    logger.info(f"Capped {num_outliers} outliers in column {col}")
        
        rows_after = len(df)
        if rows_before > rows_after:
            logger.info(f"Removed {rows_before - rows_after} rows containing outliers")
            
        return df
    
    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that values are within expected ranges and fix if needed.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with validated ranges.
        """
        for col, (min_val, max_val) in self.VALID_RANGES.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Count values outside valid range
                below_min = (df[col] < min_val).sum()
                above_max = (df[col] > max_val).sum()
                
                if below_min > 0:
                    logger.warning(f"{below_min} values in {col} below minimum ({min_val})")
                    df.loc[df[col] < min_val, col] = min_val
                
                if above_max > 0:
                    logger.warning(f"{above_max} values in {col} above maximum ({max_val})")
                    df.loc[df[col] > max_val, col] = max_val
        
        return df
    
    def _log_data_stats(self, df: pd.DataFrame, stage: str) -> None:
        """
        Log basic statistics about the dataframe.
        
        Args:
            df (pd.DataFrame): The dataframe to analyze.
            stage (str): Description of the current processing stage.
        """
        if not self.verbose:
            return
            
        logger.debug(f"Data statistics {stage}:")
        logger.debug(f"- Shape: {df.shape}")
        
        # Log stats for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                stats = df[col].describe()
                logger.debug(f"- {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
            except:
                logger.debug(f"- {col}: Could not calculate statistics")
        
        # Count missing values
        missing = df.isna().sum().sum()
        logger.debug(f"- Missing values: {missing}")

def clean_weather_data(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Convenience function to clean weather data.
    
    Args:
        data (pd.DataFrame): The weather dataframe to clean.
        verbose (bool, optional): Whether to print detailed logs. Defaults to False.
        
    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    cleaner = WeatherDataCleaner(verbose=verbose)
    return cleaner.clean_dataframe(data)

# Example usage
if __name__ == "__main__":
    # Example with sample data
    import sys
    
    # Create sample data with some issues
    sample_data = pd.DataFrame({
        'DATE': pd.date_range(start='2022-01-01', end='2022-01-10'),
        'STATION': ['S1', 'S2', 'S3', 'S4', 'S5', None, 'S7', 'S8', 'S9', 'S10'],
        'TMAX': [25, 26, 27, 90, 26, 25, 24, 23, 22, None],  # Outlier at index 3
        'TMIN': [10, 11, 12, 13, -100, 11, 10, 9, 8, 7],  # Outlier at index 4
        'PRCP': [5, 6, 7, 8, 9, 10, 11, 3000, 13, 14]  # Outlier at index 7
    })
    
    print("Sample data before cleaning:")
    print(sample_data)
    
    # Clean the data
    cleaned_data = clean_weather_data(sample_data, verbose=True)
    
    print("\nSample data after cleaning:")
    print(cleaned_data)