import yfinance as yf
import json
from typing import Any, Dict


class StockInfo:
    """
    Get information about a stock Ticker
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about a stock Ticker
        """

        try:
            self.stock = yf.Ticker(self.ticker)
            info = self.stock.info
            # Format the response
            result = {
                "symbol": self.ticker,
                "company_name": info.get("longName", "N/A"),
                "current_price": info.get("currentPrice", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", "N/A")
            }
            return result
        except Exception as e:
            return {"error": f"Error getting data for {self.ticker}: {str(e)}"}

if __name__ == "__main__":
    stock_info = StockInfo("AAPL")
    print(json.dumps(stock_info.get_info(),indent=2))