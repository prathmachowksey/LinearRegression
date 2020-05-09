# Multivariate-Linear-Regression
In this assignment, multivariate linear regression models have been developed for the [3D Road Network (North Jutland, Denmark) Data Set](https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland%2C+Denmark)) using gradient descent, stochastic gradient descent, solving normal equations and gradient descent with regularisation. R squared, RMSE and Squared Error values have been calculated and compared for each model.

## Dataset
- Number of Instances: 43487
- Number of Attributes: 4

### Attributes:
 - OSM_ID: OpenStreetMap ID for each road segment or edge in the graph.
 - LONGITUDE: (Google format) longitude
 - LATITUDE: (Google format) latitude
 - ALTITUDE: Height in meters. 
 
The first attribute(OSM_ID) has been dropped. LONGITUDE and LATITUDE values have been used to predict the target variable, ALTITUDE.
