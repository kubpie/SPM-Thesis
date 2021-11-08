# Solution to the SPM problem with an XGBoost model
<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/xgb_pipe.jpg"/>
</p>

xgb_main.py once executed runs a complete machine learning pipeline as described in the above diagram. The complete process includes comparison of a regression model and classification model due to characteristics of the target variable (more details in the report).
In the comparison the classifier model achieves better results with an overall F1-macro score of 0.72.

<p align="center">
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/xgb_results.jpg" width="100"/>
  <img src="https://github.com/kubpie/SPM-Thesis/blob/master/pics/xgb_learning_curves.jpg" width="100"/>   
</p>
