#import libraries
import pandas as pd

#open the data file
df = pd.read_csv('kdl.csv')

#get all columns for screen data
columns = []

for i in range(40):
    for j in range(4):
        columns.append(f"screen{i+1}_{j}")

#run summary stats on screen only data - quantative
print(df[columns].stack().describe(), end='\n')
print("Name: Screen Data", end='\n\n')

#get all columns for tile data
columns = []

for i in range(41):
    for j in range(4):
        columns.append(f"tile{i+1}_{j}")

#run summary stats on tile only data - quantative
print(df[columns].stack().describe(), end='\n')
print("Name: Tile Data", end='\n\n')

#run summary stats on kirby's scroll x-value data - quantative
print(df["kirby_x_scrol"].describe(), end='\n\n')

#run summary stats on kirby's scroll y-value data - quantative
print(df["kirby_y_scrol"].describe(), end='\n\n')

#run summary stats on kirby's x-value data - quantative
print(df["kirby_x"].describe(), end='\n\n')

#run summary stats on kirby's y-value data - quantative
print(df["kirby_y"].describe(), end='\n\n')

#run summary stats on boss health data - quantative
print(df["boss_health"].describe(), end='\n\n')

#run summary stats on game state data - nominal
print(df["game_state"].value_counts(), end='\n\n')

#run summary stats on move data - nominal, output/prediction variable
print(df["move"].value_counts(), end='\n\n')

