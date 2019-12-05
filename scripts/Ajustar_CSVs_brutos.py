#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:34:01 2019

11/112019. César. Obs: Antes de iniciar o script deve-se checar (no CSV) o nome das variaveis que serão usadas.
Existem casos que a temperatura está como "t+AF8-degree+AF8-celsius" já outros 
"t_degree_celsius". Então, atenção ao usar o rename e na ordenação das colunsa depois.


Novidades: Essa versão retira caracter ":" da hora, o "-" do dia, retira as colunas de velocidade
e coloca a variável que vai ser "predict" na primeira coluna. 

---->>> Maior novidade: salvas as modificações no mesmo arquivo de entrada, não precisa de outro
arquivo de output

@author: cesar
"""
from keras.models import Sequential
from sklearn.metrics import r2_score #Novo#
from keras.layers import Dense, Dropout, LSTM # LSTM adicionado na aula 70
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.layers import Bidirectional
from keras.models import load_model
from sklearn.metrics import r2_score

arquivos = ['inverno20162021.csv']
end='/home/cesar/Dropbox/Gerar resultados/'

for a in arquivos:
    base = pd.read_csv(a)
    
    #Ajustar nome de saída
    
    file = end+a
    
    
    
    #Apagar registros NaN
    base = base.dropna()
    
    #Retirar carcteres de hora
    base['hora'] = base['hora'].replace(':', '', regex=True)
    base['hora.1'] = base['hora.1'].replace(':', '', regex=True)
    base['hora.2'] = base['hora.2'].replace(':', '', regex=True)
    base['hora.3'] = base['hora.3'].replace(':', '', regex=True)
    
    #Retirar caracter de data
    base['data'] = base['data'].replace('-', '', regex=True)
    base['data.1'] = base['data.1'].replace('-', '', regex=True)
    base['data.2'] = base['data.2'].replace('-', '', regex=True)
    base['data.3'] = base['data.3'].replace('-', '', regex=True)
    
    #Arrumar nome de colunas de temperatura
    #Renomear colunas^
    #base.rename(columns={'t+AF8-degree+AF8-celsius': 't2016', 
    #                     't+AF8-degree+AF8-celsius.1': 't2011a', 
    #                     't+AF8-degree+AF8-celsius.2': 't2011b', 
    #                     't+AF8-degree+AF8-celsius.3': 't2011c'}, 
    #           inplace=True)
    
    #Arrumar nome de colunas de temperatura
    #Renomear colunas^
    base.rename(columns={'t_degree_celsius': 't2016', 
                         't_degree_celsius.1': 't2016a', 
                         't_degree_celsius.2': 't2016b', 
                         't_degree_celsius.3': 't2016c'}, 
               inplace=True)
    
    
    #Dropar colunas de velocidade
    velocidade = ['velocidade', 'velocidade.1','velocidade.2', 'velocidade.3']
    base.drop(velocidade, axis=1, inplace=True)
    #del base["velocidade", "velocidade.1","velocidade.2", "velocidade.3"]
    print(base)
    
    #Mudar ordem colocando temperaura na primeira colunas
    #base = base[['t2016','data', 'long', 'lat', 'hora', 'umidade', 'altitude',
     #      'cobpais_pct', 'cobarb_pct', 'soloexp_pct', 'apav_pct', 'aedf',
      #     'aguapct', 'data.1', 'long.1', 'lat.1', 'hora.1', 't2016a', 'umidade.1',
       #    'altitude.1', 'cobpais_pct.1', 'cobarb_pct.1', 'soloexp_pct.1',
        #   'apav_pct.1', 'aedf.1', 'aguapct.1', 'data.2', 'long.2', 'lat.2',
         #  'hora.2', 't2016b', 'umidade.2', 'altitude.2', 'cobpais_pct.2',
         #  'cobarb_pct.2', 'soloexp_pct.2', 'apav_pct.2', 'aedf.2', 'aguapct.2',
         #  'data.3', 'long.3', 'lat.3', 'hora.3', 't2016c', 'umidade.3',
         #  'altitude.3', 'cobpais_pct.3', 'cobarb_pct.3', 'soloexp_pct.3',
         #  'apav_pct.3', 'aedf.3', 'aguapct.3']]
    
    #print(base)

    #Mudar ordem colocando temperaura na primeira colunas
    #base = base[['t2016','data', 'long', 'lat', 'hora', 'umidade', 'altitude',
     #      'cobpais_pct', 'cobarb_pct', 'soloexp_pct', 'apav_pct', 'aedf',
     #      'aguapct', 'data.1', 'long.1', 'lat.1', 'hora.1', 't2016a', 'umidade.1',
     ##      'altitude.1', 'cobpais_pct.1', 'cobarb_pct.1', 'soloexp_pct.1',
     #      'apav_pct.1', 'aedf.1', 'aguapct.1', 'data.2', 'long.2', 'lat.2',
     #      'hora.2', 't2016b', 'umidade.2', 'altitude.2', 'cobpais_pct.2',
     #      'cobarb_pct.2', 'soloexp_pct.2', 'apav_pct.2', 'aedf.2', 'aguapct.2',
     #      'data.3', 'long.3', 'lat.3', 'hora.3', 't2016c', 'umidade.3',
     #      'altitude.3', 'cobpais_pct.3', 'cobarb_pct.3', 'soloexp_pct.3',
     #      'apav_pct.3', 'aedf.3', 'aguapct.3']]

    #Salvar em CSV após todas modificações
    #base.to_csv(r'/home/cesar/novo.csv')
    #print(base)
    
    #Salvar em CSV após todas modificações
    base.to_csv(file, index = False)
    print(base)
print('ja acabou \o/')
