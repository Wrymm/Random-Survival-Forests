from survival import RandomSurvivalForest
import pandas as pd
import numpy as np

data = pd.DataFrame(np.array([[1,3,10,1],[2,7,1,0],[8,2,2,1],[7,3,12,0],[8,6,30,1],[1,8,11,1],[4,4,10,1],[6,8,28,1]]), columns=["f1","f2","time","e"])

rd = RandomSurvivalForest(n_trees=2)
rd.fit(data[["f1","f2"]], data[["time","e"]])
pred = rd.predict_proba(data[["f1","f2","time"]])
print(pred)
rd.draw()
