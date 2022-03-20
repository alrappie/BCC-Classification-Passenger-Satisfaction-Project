import numpy as np
from collections import Counter

class Scratch_KNearestClassification():

    def __init__(self,n_neighbors:int) -> None:
        self.n_neighbors = n_neighbors
        self.output = np.array([])
        
    #fit data frame type
    def fit(self,x,y):
        self.data_fitted = np.array(x.values)
        self.output = np.array(y)
        return print("Scratch_KNearestClassification()")
    
    def predict(self,input):
        input = np.array(input)
        outputAkhir = []
        for i in range(input.shape[0]):
            array_ED = []
            for k in range(self.data_fitted.shape[0]):
                hasil = np.sqrt(np.sum(np.power((self.data_fitted[k]-input[i]),2)))
                array_ED.append(hasil)
            index = np.array(array_ED).argsort()[:self.n_neighbors]
            label = self.output[index]
            counter = Counter(label)
            kelas_uji = counter.most_common(1)[0][0]
            outputAkhir.append(kelas_uji)
        return np.array(outputAkhir)