"""
Role:
 - Use yfinance library to download stock price data
 - Conduct Basic Data Preprocessing
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta


class DataDownloader:
    """
    Parameters
    ----------
    symbol : str
        e.g) 'SPY', 'AAPL'
    start_date : str
        default : 5 years from now
    end_date : str
        default : today
    """

    def __init__(
        self,
        symbol: str,
        start_date: str = (datetime.now() - relativedelta(days=365 * 5)).strftime("%Y-%m-%d"),
        end_date: str = datetime.now().strftime("%Y-%m-%d"),
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def download_from_yf(
        self,
        group_by: str = "tickers",
        auto_adjust: bool = True,
        progress: bool = False,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        group_by : str
            yfinance download option. Common: 'tickers' or 'column'
        auto_adjust : bool
            adjust OHLC for splits/dividends
        progress : bool
            show progress bar

        Returns
        -------
        pd.DataFrame
            OHLCV with lowercase columns
        """

        temp_df = (
            yf.download(
                tickers=self.symbol,
                start=self.start_date,
                end=self.end_date,
                group_by=group_by,
                auto_adjust=auto_adjust,
                progress=progress,
            )
        )

        # ✅ yfinance가 ticker를 level로 주는 경우(특히 여러 ticker or group_by 옵션)에만 droplevel
        #    단일 ticker일 때는 columns가 그냥 Index일 수 있어서 안전하게 처리
        if isinstance(temp_df.columns, pd.MultiIndex):
            # 보통 0레벨이 ticker, 1레벨이 OHLCV
            # 어떤 경우는 반대로 올 수 있어 안전하게 "ticker가 있는 레벨"을 찾기 어려우니
            # 일단 가장 흔한 케이스인 level=0 droplevel 시도
            try:
                temp_df = temp_df.droplevel(axis=1, level=0)
            except Exception:
                # fallback: level=1로도 한번 시도
                temp_df = temp_df.droplevel(axis=1, level=1)

        temp_df = (
            temp_df.rename(columns=str.lower)
            .sort_index()
        )

        # ✅ 완전 비어있는 행 제거(시장 휴일 등으로 생길 수 있음)
        temp_df = temp_df.dropna(how="all")

        # ✅ index가 timezone-aware로 들어오면 제거(후속 처리 안정화)
        try:
            temp_df.index = temp_df.index.tz_localize(None)
        except Exception:
            pass

        return temp_df

