import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path


class IndexModel:
    """Calculates index level and exports it to a .csv file."""

    def __init__(self) -> None:
        """Constructor reads the csv and exposes it to class methods as a pandas data frame.

        stock_prices_csv_df = input timeseries pricing information for index construction.
        """
        stock_prices_csv = Path(__file__).parent / "../data_sources/stock_prices.csv"
        self.stock_prices_csv_df = pd.read_csv(
            stock_prices_csv, parse_dates=["Date"], infer_datetime_format=True
        )
        self.index_vector = None

    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> None:
        """Calculates index level.

        Args:
            start_date: index_vector inception
            end_date: Last date for index value calculation
        """
        self.stock_prices_csv_df["year"] = self.stock_prices_csv_df["Date"].dt.year
        self.stock_prices_csv_df["month"] = self.stock_prices_csv_df["Date"].dt.month
        self.stock_prices_csv_df["month_year"] = (
            self.stock_prices_csv_df["Date"].dt.strftime("%Y%m").astype(str)
        )

        self.stock_prices_csv_df.set_index(["Date"], drop=False, inplace=True)
        self.stock_prices_csv_df.sort_index(inplace=True)

        # The index business days are Monday to Friday
        self.stock_prices_csv_df = self.stock_prices_csv_df[
            self.stock_prices_csv_df.index.dayofweek < 5
        ]

        # Find the last trading day in each review month and subsequently find the 3 largest market caps
        # use these selected constituents and run throughout the month.
        rebalancing_slice = self.stock_prices_csv_df.groupby("month_year")
        index_constituents = pd.DataFrame()
        rebalancing_constituents = list()
        border_days = []

        inception = True

        for month_index in rebalancing_slice.groups:
            rebalancing_month = rebalancing_slice.get_group(month_index)
            rebalancing_month.set_index(["Date"], inplace=True)
            rebalancing_month = rebalancing_month.drop(
                columns=["year", "month", "month_year"]
            )

            # The selected stock with the highest market capitalization gets assigned a 50% weight,
            # while the second and third each get assigned 25%. 3 largest constituents which are selected
            # in the previous review period is then weighted at 50%,25%,25% throughout the month

            if inception is True:
                start_i = rebalancing_month.index[0]
                end_i = rebalancing_month.index[-1]
                border_days.append([start_i, end_i])

            else:
                rebalancing_month_copy = rebalancing_month.copy()
                rebalancing_month_copy[rebalancing_constituents] = 1
                index_constituents = index_constituents.append(
                    rebalancing_month_copy[rebalancing_constituents] * [50, 25, 25],
                    ignore_index=False,
                )
                start_i = rebalancing_month_copy.index[0]
                end_i = rebalancing_month_copy.index[-1]
                border_days.append([start_i, end_i])

            rebalancing_month["1st Largest"] = rebalancing_month.T.apply(
                lambda x: x.nlargest(1).idxmin()
            )
            rebalancing_month["2nd Largest"] = rebalancing_month.drop(
                columns=["1st Largest"]
            ).T.apply(lambda x: x.nlargest(2).idxmin())
            rebalancing_month["3rd Largest"] = rebalancing_month.drop(
                columns=["1st Largest", "2nd Largest"]
            ).T.apply(lambda x: x.nlargest(3).idxmin())

            # Every first business day of a month the index selects from the universe the top three stocks based
            # on their market capitalization, based on the close of business values as of the last business day
            # of the immediately preceding month.
            rebalancing_constituents = list(
                rebalancing_month[["1st Largest", "2nd Largest", "3rd Largest"]].iloc[
                    -1
                ]
            )

            inception = False

        # The selection becomes effective close of business on the first business date of each month.
        for index in range(1, len(border_days) - 1):
            index_constituents.loc[border_days[index + 1][0]] = index_constituents.loc[
                border_days[index][1]
            ]

        multiplier = self.stock_prices_csv_df.filter(like="Stock")
        multiplier = multiplier.pct_change()

        self.index_vector = index_constituents * multiplier

        # The index start date is January 1st 2020.
        self.index_vector = self.index_vector[
            (self.index_vector.index >= pd.to_datetime(start_date))
            & (self.index_vector.index <= pd.to_datetime(end_date))
        ]
        self.index_vector["return"] = self.index_vector.sum(axis=1) / 100
        self.index_vector["IV"] = 100 * np.exp(
            np.nan_to_num(self.index_vector["return"].cumsum())
        )

        # normalise, The index starts with a level of 100.
        self.index_vector["index_level"] = round(
            self.index_vector["IV"] * 100 / self.index_vector["IV"][0], 2
        )

    def export_values(self, file_name: str) -> None:
        """Exports index vector to csv

        Args:
            file_name: file name with .csv extension
        """

        self.index_vector.reset_index(drop=False, inplace=True)
        self.index_vector[["Date", "index_level"]].to_csv(file_name, index=False)
