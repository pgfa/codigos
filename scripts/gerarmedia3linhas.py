# Pegar script a cada 3 linhas
import numpy as np
import pandas as pd

# test data
arquivo = ['inverno2016.csv', 'outono2016.csv', 'primavera2016.csv', 'verao2016.csv']
for z in arquivo:
    nome = pd.read_csv(z)
    print(nome)
    j = np.array(nome)
    media = np.mean(j.reshape(-1, 3), axis=1)
    np.savetxt('Media.'+z, media, fmt="%f", delimiter=",")