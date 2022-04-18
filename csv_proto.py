import csv

with open("ratings.csv") as file:
    data = list(csv.reader(file))
    for i in range(len(data)):
        data[i] = data[i][3:]
        print(data[i])