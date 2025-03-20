

import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
    'Age': [25, 30, 22, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney'],
    'Salary': [60000, 75000, 55000, 90000, 70000]
}

df = pd.DataFrame(data)
print(df)

