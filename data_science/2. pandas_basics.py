import pandas as pd

cars = pd.read_csv('assets/cars.csv', index_col=0)

# Print out country column as Pandas Series
print(cars['country'], end='\n\n')

# Print out country column as Pandas DataFrame
print(cars[['country']], end='\n\n')

# Print out DataFrame with country and drives_right columns
print(cars[['country', 'drives_right']], end='\n\n')

# Print out first 4 observations
print(cars[0:4], end='\n\n')

# Print out fifth and sixth observation
print(cars[4:6], end='\n\n')

# Print out observation for Japan
print(cars.iloc[2], end='\n\n')

# Print out observations for Australia and Egypt
print(cars.loc[['AUS', 'EG']], end='\n\n')
