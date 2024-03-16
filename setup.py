
import sqlite3
import glob


sqliteConnection = sqlite3.connect('data.db')
cursor = sqliteConnection.cursor()
print("\nSucessfully Connected to the database...\n")


sql_command = """CREATE TABLE fine_tuned ( 
BaseNetwork TEXT,
Accuracy FLOAT,
F1Score FLOAT,
Loss FLOAT,
AUCValue FLOAT);"""

cursor.execute(sql_command)


sqliteConnection.commit()

sqliteConnection.close()

import sys
sys.exit()
sql_command = """CREATE TABLE optimal_hyp ( 
BaseNetwork TEXT,
ActivationFunction TEXT,
HiddenUnits INT,
HiddenLayers INT,
LearningRate FLOAT);
"""
 
# execute the statement
cursor.execute(sql_command)

sql_command = """CREATE TABLE model_metrics ( 
BaseNetwork TEXT,
Accuracy FLOAT,
F1Score FLOAT,
Loss FLOAT,
AUCValue FLOAT);"""
 
# execute the statement
cursor.execute(sql_command)



sqliteConnection.commit()

sqliteConnection.close()
print("\nCreated Data Tables...")