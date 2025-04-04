import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes

# x = [i for i in range(10)]

# y = [2*i for i in range(10)]


# plt.xlabel('x-axis')
# plt.ylabel('y-axis')

# plt.bar(x, y)
# plt.show()

X, y = load_diabetes(return_X_y=True)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#mod = KNeighborsRegressor() #using the regressor model, then fitting it
#mod.fit(X, y)
#pred = mod.predict(X)

pipe = Pipeline([
    ("scale", StandardScaler()), # scaling helps no single feature dominates the distance calculations in an algorithm, helping better the algorithm
    ("model", KNeighborsRegressor())# the model for my prediction, the base neighbors it finds is 5
])

mod = GridSearchCV(estimator=pipe,
             param_grid={'model__n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},#check all then_neighbors 1 - 10 #by using pipe.get_params, youll get all the parameters you can tweak
             cv = 3)#cross validation param is 3 
mod.fit(X, y)
cv_results = pd.DataFrame(mod.cv_results_)# this gives us a dataframe about the results, which one is best etc

#plt.scatter(mod, y)# in this plot, the x-value (the prediction) says chance of diabetes around 200, it generally close and it is close as the y is what it actually is
#plt.show()#the values here represent the severity of diabetes. higher is worse
