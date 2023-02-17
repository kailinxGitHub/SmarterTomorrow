import yfinance as yf
symbol = input(str("Company Ticker Symbol (aka Stock Name)"))

ticker_symbol = yf.Ticker(symbol)
company_name = ticker_symbol.info['longName']
print(company_name)



