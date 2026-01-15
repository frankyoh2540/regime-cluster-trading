"""
Role:
 - Scale features (fit on train only)
 - Apply PCA (fit on train only)
 - Transform train/test features into PCA space
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA


class PreprocessPCA:
    # Scaler_Registry
    _SCALERS = \
    (
        {
            "standard": StandardScaler,
            "std": StandardScaler,
            "robust": RobustScaler,
            "minmax": MinMaxScaler,
            "min_max": MinMaxScaler
        }
    )
    
    def __init__(self, pca_var: float = 0.95, scaler_type: str = "standard"):
        self.pca_var = pca_var
        self.scaler_type = scaler_type
        self.scaler = None
        self.pca = None

    def _make_scaler(self, scaler_type: str = None):
        st = self.scaler_type
        st = st.lower().strip()

        if st not in self._SCALERS:
            valid = sorted(set(self._SCALERS.keys()))
            raise ValueError(f"Unknown scaler_type='{st}'. Use one of: {valid}")

        return self._SCALERS[st]()

    def fit(self, X_train: pd.DataFrame, scaler_type: str = None):
        """
        Fit scaler + PCA on train only
        scaler_type: standard, robust, minmax
        """
        self.scaler = self._make_scaler(scaler_type)
        Z = self.scaler.fit_transform(X_train.values)

        self.pca = PCA(
            n_components=self.pca_var,
            svd_solver="full",
            random_state=42
        )
        self.pca.fit(Z)

        return self
    

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform any X into PCA space using fitted scaler + PCA
        """
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Call fit(X_train) first.")

        Z = self.scaler.transform(X.values)
        Y = self.pca.transform(Z)

        cols = [f"pc{i+1}" for i in range(Y.shape[1])]
        return pd.DataFrame(Y, index=X.index, columns=cols)

    def fit_transform(self, X_train: pd.DataFrame, scaler_type: str = None) -> pd.DataFrame:
        self.fit(X_train, scaler_type=scaler_type)
        return self.transform(X_train)
