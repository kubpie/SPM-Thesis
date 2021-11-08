# Solution to the SPM problem with an XGBoost model
<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/xgb_pipe.jpg"/>
</p>

**xgb_main.py** once executed runs a complete machine learning pipeline as described in the above diagram. The complete process includes comparison of a regression model and classification model due to characteristics of the target variable (more details in the report).
In the comparison the classifier model achieves better results with an overall F1-macro score of 0.72.

<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/xgb_results.JPG" width = 250/>
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/xgb_learning_curves.JPG" width = 550/>   
</p>

It has been made clear in data analysis that simple data manipulation can lead to splitting the problem into 3 smaller sub-problems with separate datapoints. The predictions made on this subsets have better prediction score on average. The models for these subsets are trained in **xgb_split_model_tuning.py** </br>
In general it has been proven that making a prediction more accurate would require generating more datapoints for the most difficult propagation scenarios. 
