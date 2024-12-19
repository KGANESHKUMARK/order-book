import pandas as pd
import matplotlib.pyplot as plt

# Sample market data (timestamp, price, volume)
data = [
    {"timestamp": "2024-12-19 10:00:00", "price": 100.5, "volume": 150},
    {"timestamp": "2024-12-19 10:01:00", "price": 101.0, "volume": 200},
    {"timestamp": "2024-12-19 10:02:00", "price": 99.8, "volume": 180},
]

# Load data into a DataFrame
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Calculate a simple moving average (SMA)
df['SMA'] = df['price'].rolling(window=2).mean()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['price'], label='Price')
plt.plot(df.index, df['SMA'], label='SMA (2)', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('Market Data Visualization')
plt.legend()
plt.grid()
plt.show()
