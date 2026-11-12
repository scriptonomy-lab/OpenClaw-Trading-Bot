"""Price code aliases for renamed stocks (D36).

Membership gives the code a stock had at the time, such as FB. EODHD serves
prices, splits and dividends for a renamed stock only under its current code,
such as META. This map says which code to ask for. It is used for prices,
splits and dividends only. Membership and news keep the original code: EODHD
tags old articles with the code of the day, so FB has 2018 news and META does not.

Every entry was checked with a live call that showed matching price history.
UA-C is not listed on purpose: it is class C stock and UA is class A.
"""

from datetime import date

# old membership code -> EODHD price code, with the company and rename date.
PRICE_ALIASES: dict[str, str] = {
    "ACE": "CB",  # ACE Limited took the Chubb name in January 2016.
    "FB": "META",  # Facebook became Meta in June 2022.
    "LB": "BBWI",  # L Brands became Bath & Body Works in August 2021.
    "MHFI": "SPGI",  # McGraw Hill Financial became S&P Global in April 2016.
    "PCLN": "BKNG",  # Priceline became Booking Holdings in February 2018.
    "Q": "IQV",  # Quintiles IMS became IQVIA in November 2017.
    "RE": "EG",  # Everest Re became Everest Group in July 2023.
}


# Last day each old code is valid as a membership code. These are the last
# days of the old spells in the cached membership file. A spell for the same
# code that starts later is a different company (a reused code, such as LB for
# LandBridge). It must not be priced with the successor's series.
ALIAS_LAST_DAY: dict[str, date] = {
    "ACE": date(2016, 1, 15),
    "FB": date(2022, 6, 8),
    "LB": date(2021, 8, 2),
    "MHFI": date(2016, 4, 27),
    "PCLN": date(2018, 2, 26),
    "Q": date(2017, 11, 14),
    "RE": date(2023, 7, 9),
}


def price_code(ticker: str) -> str:
    """The code to ask EODHD for prices, splits and dividends of a membership code."""
    return PRICE_ALIASES.get(ticker, ticker)


def company_key(symbol: str) -> str:
    """A key that is the same for a renamed stock's old and new symbol, such as FB.US and META.US.

    Used to stop the book from holding one company twice across a rename.
    """
    code, dot, exchange = symbol.partition(".")
    return f"{price_code(code)}{dot}{exchange}"
