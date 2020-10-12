import sweetviz as sv
odata_day = pd.read_csv("data/input/Bike-Sharing-Dataset/day.csv")

data_day = odata_day.copy()
data_day['date'] = pd.to_datetime(data_day['dteday']).dt.floor('d')
data_day.drop(columns='dteday', inplace=True)
data_day = data_day.set_index('date')
cat_column_names = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
data_day.loc[:, "season":"weathersit"] = data_day.loc[:, "season":"weathersit"].astype("category") 

my_report = sv.analyze(data_day, "cnt", sv.FeatureConfig(skip="instant"))
my_report.show_html("initial_eda.html")

f = plt.figure()
plt.plot(data_day_2011.index, data_day_2011['cnt'], marker='.', alpha=0.5, linestyle='None')
plt.plot(data_day_2011.index, data_day_2012['cnt'].drop(index=365), marker='.', alpha=0.5, linestyle='None')




cols_plot = ['cnt', 'casual', 'registered']
axes = data_day[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily Totals')

np.concatenate((X_train, catVar), axis = 1)






cols_plot = ['cnt', 'casual', 'registered']
axes = data_train[cols_plot].plot(linewidth=0.5, subplots=False)





RandomForestRegressor(n_estimators=100, min_samples_leaf=0.01, max_features=0.8, bootstrap=True, max_samples=1, oob_score=True, random_state=123, verbose=0, n_jobs=-1)


# RMeanSE
# MeanAE
# MedianAE


def get_eval_metric(y_true, y_pred, y_pred_baseline="auto"):
    error = y_pred - y_true
    abs_error = np.abs(error)
    squared_error = np.square(error)

    if y_pred_baseline == "auto":
        abs_error_baseline = np.abs(y_pred_baseline - y_true)
        MAE_r2 = 1 - (np.mean(abs_error)/np.mean(abs_error_baseline))
    else:
        abs_error_baseline = np.abs(y_pred_baseline - y_true)
        MAE_r2 = 1 - (np.mean(abs_error)/np.mean(abs_error_baseline))

    return {
        'MAE': np.mean(abs_error),
        'MAPE': np.mean(abs_error_percent),
        'median_PE': np.median(abs_error),
        'median_APE': np.median(abs_error_percent),
        'MAE_r2': MAE_r2


def scaled_RMSE(y_true, y_pred):
    """Standard deviation adjusted mean absolute error regression loss
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    loss : float
        MAE output is non-negative floating point. The best value is 0.0.
    """
    
    abs_error = np.abs(y_pred - y_true)
    scaled_RMSE = np.mean(abs_error)/np.std(abs_error)

    return scaled_RMSE

my_scaled_RMSE_scorer = make_scorer(scaled_RMSE, greater_is_better=False)



for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, final_selected_columns[indices[f]], importances_normalized[indices[f]]))



base RF
               RMeanSE  MeanAE  MedianAE  RMeanSE_r2  MeanAE_r2  MedianAE_r2
train_metrics   257.61  181.84     132.0      0.8685     0.8864       0.9111
              RMeanSE  MeanAE  MedianAE  RMeanSE_r2  MeanAE_r2  MedianAE_r2
test_metrics  1053.57   900.0     895.5      0.1582     0.1884       0.0908


Tuned RF
               RMeanSE  MeanAE  MedianAE  RMeanSE_r2  MeanAE_r2  MedianAE_r2
train_metrics   603.85  437.81     327.0      0.6917     0.7265       0.7799
              RMeanSE  MeanAE  MedianAE  RMeanSE_r2  MeanAE_r2  MedianAE_r2
test_metrics  1016.43   885.9     871.5      0.1879     0.2012       0.1152

data_beforeLast30Days_2011 = data_beforeLast30Days.loc[data_beforeLast30Days.yr == 0,:]
data_beforeLast30Days_2012 = data_beforeLast30Days.loc[data_beforeLast30Days.yr == 1,:]
data_beforeLast30Days_2012 = data_beforeLast30Days_2012.reset_index(drop = True).drop(index=365)

for target_var in ['cnt', 'casual', 'registered']:
    f = plt.figure()
    plt.plot(data_beforeLast30Days_2011.index, data_beforeLast30Days_2011[target_var], '-')
    plt.plot(data_beforeLast30Days_2011.index, data_beforeLast30Days_2012[target_var], '--')
    plt.show()





* Lst 30 days  is last 31 days actually    


seed=np.random.randint(100, size=1).item()



To do:
- ts features
- casual + registered
- plot test vs last 30 days errors

- 30 days of data is actually 31 days of data!. I just took the last month, and then it was too much effort to make it 30 and rename part of variable names from 30 to 31
