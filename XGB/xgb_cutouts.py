
### Pipeline cutouts

"""
it = 0
for param in [param4]:
    it = it + 1
    r_res, r_param = HyperParamGS(xgb_reg, param, dtrain, features, target, scoring_reg, refit = 'F1-macro')
    dump(r_res, resultpath + "reg_param_L1" + str(it) + ".dat")
    xgb_reg = xgb_reg.set_params(**r_param)
    PlotGS(r_res, param, scoring_reg, modeltype='reg_L1')

dump(xgb_reg, "L1_reg.dat")

it = 0
for param in [param4]:
    it = it + 1
    c_res, c_param = HyperParamGS(xgb_class, param, dtrain, features, target, scoring_class, refit = 'F1-macro')
    dump(c_res, resultpath + "class_param_L1" + str(it) + ".dat")
    xgb_class = xgb_class.set_params(**c_param)
    PlotGS(c_res, param, scoring_class, modeltype='class_L1')

dump(xgb_class, "L1_fine.dat")


#param = param3
#class_res, class_param = HyperParamGS(xgb_class, param, dtrain, features, target, scoring_class, refit = 'F1-macro')
#dump(class_res, resultpath + "param4_class.dat")
#xgb_class = xgb_class.set_params(**class_param)
#PlotGS(class_res, param, scoring_class)
"""

param_test = {
    'learning_rate': 0.05,
    'n_estimators': 2000
}
#xgb_reg = load(resultpath+"coarse_reg.dat")
#xgb_class = load(resultpath+"coarse_class.dat")

#xgb_class = xgb_class.set_params(**param_test)
#xgb_reg = xgb_reg.set_params(**param_test)

##### TRAIN THE BEST MODEL ON THE FULL DATASET AND FIT ON THE TEST SET #####
bst_model_class, fullresult_class, output_class = ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100, verbose=True, learningcurve = True, importance = True, plottree = False, savename = "class_feat")
#bst_model_reg, fullresults_reg, output_reg  = ModelFit(xgb_reg, dtrain, dtest, features, target, early_stop = 200, verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)

"""
#from imblearn.over_sampling import SMOTENC
# TODO: Make a function out of this too

# Upsampling with SMOT-ENC technique that can handle both cont. and categorical variables
#categorical_var = np.hstack([2, np.arange(5,33)])
categorical_var = np.hstack([2,np.arange(5,33)])
minority = np.arange(4,17)
samplenr = 250
population_target = dict(zip(minority, (np.ones(len(minority))*samplenr).astype(int)))
smote_nc = SMOTENC(categorical_features=categorical_var, sampling_strategy=population_target, random_state=42)
#smote_nc_max = SMOTENC(categorical_features=categorical_var, sampling_strategy='auto', random_state=42)
X_smot, y_smot = smote_nc.fit_resample(X_train, y_train)
dtrain_smot = pd.concat((X_smot, y_smot), axis =1)
dtrain_smot = dtrain_smot.sample(frac = 1) #shuffle the upsampled dataset
"""
"""
def TrainTestSplit(data, save = False, seed = 27, test_size = 0.25):
    # divide dataset into test & training subsets
    target = 'num_rays'
    predictors = [x for x in data.columns if x not in target]
    X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=test_size, random_state=seed, stratify =  data[target])
    # stratified split ensures that the class distribution in training\test sets is as similar as possible
    dtrain = pd.concat((X_train, y_train), axis = 1)
    dtest = pd.concat((X_test, y_test), axis = 1)
    
    if save:
        # save into separate .csv files
        filepath = os.getcwd()+'\data\\'
        dtest.to_csv(filepath + 'dtest_25.csv', index = None, header = True)
        dtrain.to_csv(filepath + 'dtrain_75.csv', index = None, header = True)
        #dtrainup.to_csv(filepath + 'dtrainup.csv', index = None, header = True)
        #dtrain_smot.to_csv(filepath + 'dtrain_smot.csv', index = None, header = True)
        print("New datafiles have been created!")
    
    return dtrain, dtest

"""