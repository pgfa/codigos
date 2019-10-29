# -*- coding: utf-8 -*-
"""
Criado em 

@author: César
"""

import pandas as pd
import matplotlib.pyplot as plt

nome = 'verao20112016.r2.csv'
base = pd.read_csv(nome)

#Pegar dados para gerar gráficos
combinacoes = base.iloc[:,0:1].values # Coluna Todas as colunas
r2 = base.iloc[:,7:8].values #Pega somente a coluna 1
plt.plot(combinacoes, r2, color = 'blue')
plt.scatter(combinacoes, r2, color="red")
plt.title('Gráfico com o resultados de R² dos modelos criados')
plt.xlabel('Combinação')
plt.ylabel('R Quadrado')
#plt.ylim([0.8, 1])
plt.grid(True)
 # Fixar tamanho do eixo Y
plt.legend()
plt.savefig(nome+'.png')
plt.show()
plt.close()
