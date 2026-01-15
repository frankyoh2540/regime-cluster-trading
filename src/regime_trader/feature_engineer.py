"""
Role:
 - Import Raw Data(OHLCV) from Downloader
 - Create Features from noise reduced price data(e.g. SMA)
 - Preprocess these data to use as input for Regime Clustering
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

class FeatureEngineer:
    def __init__(
        self,
        price_col: str = "close",
        reduce_price_window: int = 5,
        feature_window: int = 30
    ):
        self.raw_price_col = price_col
        self.reduced_price_window = reduce_price_window
        self.feature_window = feature_window

   #  Functions For Basic Calculation"

    def _calc_noise_reduced_price(self, df: pd.DataFrame) -> pd.Series:
        return df[self.raw_price_col].rolling(self.reduced_price_window).mean()

    def _calc_return(self, sr: pd.Series) -> pd.Series:
        return sr.pct_change()

    def _calc_vol(self, sr: pd.Series) -> pd.Series:
        return sr.rolling(self.feature_window).std()

    def _calc_skew(self, sr: pd.Series) -> pd.Series:
        return sr.rolling(self.feature_window).apply(lambda x: skew(x), raw=True)

    def _calc_excess_kurt(self, sr: pd.Series) -> pd.Series:
        return sr.rolling(self.feature_window).apply(lambda x: kurtosis(x, fisher=True), raw=True)



    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline:
        df(OHLCV) -> noise reduced price(eg. SMA) -> return -> rolling stats -> feature df
        """

        # 1) noise reduced price (SMA)
        nrp = self._calc_noise_reduced_price(df)

        # 2) return based on noise reduced price
        ret = self._calc_return(nrp)

        # ===========================
        # 1st-order features (ret stats)
        # ===========================
        vol_ret = self._calc_vol(ret)              # vol of ret
        skew_ret = self._calc_skew(ret)            # skew of ret
        kurt_ret = self._calc_excess_kurt(ret)     # kurt of ret(excess)

        # ===========================
        # 2nd-order features (stats of stats)
        # ===========================
        vol_of_vol = self._calc_vol(vol_ret)
        skew_of_vol = self._calc_skew(vol_ret)
        kurt_of_vol = self._calc_excess_kurt(vol_ret)

        vol_of_skew = self._calc_vol(skew_ret)
        vol_of_kurt = self._calc_vol(kurt_ret)
        skew_of_kurt = self._calc_skew(kurt_ret)

        # ===========================
        # merge
        # ===========================
        feat = pd.concat(
            [
                nrp,
                ret,
                vol_ret,
                skew_ret,
                kurt_ret,
                vol_of_vol,
                skew_of_vol,
                kurt_of_vol,
                vol_of_skew,
                vol_of_kurt,
                skew_of_kurt,
            ],
            axis=1
        )

        feat.columns = [
            "nr_price",
            "ret",
            "vol_of_ret",
            "skew_of_ret",
            "kurt_of_ret_excess",
            "vol_of_vol",
            "skew_of_vol",
            "kurt_of_vol_excess",
            "vol_of_skew",
            "vol_of_kurt",
            "skew_of_kurt",
        ]

        return feat.dropna()

