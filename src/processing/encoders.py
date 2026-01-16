# src/processing/encoders.py
"""Categorical and numeric field encoding utilities.

This module provides encoders for transforming categorical and numeric
metadata fields into vector representations suitable for similarity computation.
"""

from typing import Optional

import numpy as np
import pandas as pd


class CategoricalEncoder:
    """Encode categorical metadata fields.

    Supports one-hot encoding and label encoding for categorical fields
    like organism, assay type, and biosample.

    Example:
        >>> encoder = CategoricalEncoder()
        >>> encoder.fit(df["assay_term_name"])
        >>> encoded = encoder.transform(df["assay_term_name"])
    """

    def __init__(self, encoding_type: str = "onehot") -> None:
        """Initialize the categorical encoder.

        Args:
            encoding_type: Type of encoding ("onehot" or "label").
        """
        self.encoding_type = encoding_type
        self._categories: Optional[list] = None
        self._fitted = False

    def fit(self, series: pd.Series) -> "CategoricalEncoder":
        """Fit the encoder to the data.

        Args:
            series: Pandas Series containing categorical values.

        Returns:
            Self for method chaining.
        """
        raise NotImplementedError("fit not yet implemented")

    def transform(self, series: pd.Series) -> np.ndarray:
        """Transform categorical values to encoded vectors.

        Args:
            series: Pandas Series containing categorical values.

        Returns:
            NumPy array with encoded values.

        Raises:
            ValueError: If encoder has not been fitted.
        """
        raise NotImplementedError("transform not yet implemented")

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            series: Pandas Series containing categorical values.

        Returns:
            NumPy array with encoded values.
        """
        return self.fit(series).transform(series)


class NumericEncoder:
    """Normalize and encode numeric metadata fields.

    Supports standardization (z-score) and min-max normalization
    for numeric fields like replicate count and file count.

    Example:
        >>> encoder = NumericEncoder(method="standardize")
        >>> encoder.fit(df["replicate_count"])
        >>> normalized = encoder.transform(df["replicate_count"])
    """

    def __init__(self, method: str = "standardize") -> None:
        """Initialize the numeric encoder.

        Args:
            method: Normalization method ("standardize" or "minmax").
        """
        self.method = method
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._fitted = False

    def fit(self, series: pd.Series) -> "NumericEncoder":
        """Fit the encoder to the data.

        Args:
            series: Pandas Series containing numeric values.

        Returns:
            Self for method chaining.
        """
        raise NotImplementedError("fit not yet implemented")

    def transform(self, series: pd.Series) -> np.ndarray:
        """Transform numeric values to normalized vectors.

        Args:
            series: Pandas Series containing numeric values.

        Returns:
            NumPy array with normalized values.

        Raises:
            ValueError: If encoder has not been fitted.
        """
        raise NotImplementedError("transform not yet implemented")

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            series: Pandas Series containing numeric values.

        Returns:
            NumPy array with normalized values.
        """
        return self.fit(series).transform(series)
