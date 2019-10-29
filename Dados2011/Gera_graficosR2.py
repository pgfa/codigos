# -*- coding: utf-8 -*-
"""
Criado em 

@author: César
"""

import pandas as pd
import matplotlib.pyplot as plt

nome = 'LSTM_inverno32011.csv'
base = pd.read_csv(nome)

#Pegar dados para gerar gráficos
t2011 = base.iloc[:,0:1].values # Coluna Todas as colunas
t2016 = base.iloc[:,1:2].values
#t2021 = base.iloc[:,2:3].values
plt.plot(t2011, color = 'blue', label = 'Dados 2011')
plt.plot(t2016, color = 'red', label = 'Dados 2016')
#plt.plot(t2021, color = 'red', label = 'Dados 2021')

plt.title('Gráfico com Temperaturas do Inverno de 2011 e 2016')
plt.xlabel('Timestamps')
plt.ylabel('Temperatura')
plt.grid(True)
 # Fixar tamanho do eixo Y
plt.legend()
plt.savefig(nome+'.png')
plt.show()
plt.close()
