import pandas as pd
import psycopg2
import sys
from tqdm import tqdm

psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)

#Estabelece conexão ao banco de dados
connect = psycopg2.connect(host="127.0.0.1", port="5432", database="TESTE", user="postgres", password="qwe123@ic")
cursor = connect.cursor()

#query para coletar os dados de 2011_ids
query2011_Data = """
Select p.*
from inverno_solo_32011_ids p

"""

#query para buscar cada ponto proximo das colunas d1, d2 e d3 de 2011_ids
queryNearPts = """
Select n.*
from inverno_solo_2011 n
where n.id = {}

"""

#Trabalha os dados e deixa somente os utilizados para a entrada da rede neural
cursor.execute(query2011_Data)
data2011 = cursor.fetchall()
dataFrameOutput = []
for row in tqdm(data2011):
    *_ , d1, d2, d3 = row
    row = row[1:len(row)-3]
    nearPointsId = [d1, d2, d3] 
    for point in nearPointsId:
        cursor.execute(queryNearPts.format(str(point)))
        result = cursor.fetchall()[0]
        result = result[1:]
        row = row + result

    dataFrameOutput.append(pd.DataFrame([row], columns=['data',
'long',
'lat',
'hora',
't_degree_celsius',
'umidade',
'altitude',
'velocidade',
'cobpais_pct',
'cobarb_pct',
'soloexp_pct',
'apav_pct',
'aedf',
'aguapct',
'data',
'long',
'lat',
'hora',
't_degree_celsius',
'umidade',
'altitude',
'velocidade',
'cobpais_pct',
'cobarb_pct',
'soloexp_pct',
'apav_pct',
'aedf',
'aguapct',
'data',
'long',
'lat',
'hora',
't_degree_celsius',
'umidade',
'altitude',
'velocidade',
'cobpais_pct',
'cobarb_pct',
'soloexp_pct',
'apav_pct',
'aedf',
'aguapct',
'data',
'long',
'lat',
'hora',
't_degree_celsius',
'umidade',
'altitude',
'velocidade',
'cobpais_pct',
'cobarb_pct',
'soloexp_pct',
'apav_pct',
'aedf',
'aguapct']))

#Empacota os dados e os salva em formato .csv
df_pontos = pd.concat(dataFrameOutput).reset_index(drop=True)
df_pontos.to_csv('LSTM_inverno32011.csv', index=False)



