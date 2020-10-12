import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,
                     use_columns=False, xticks=None, colormap=None,
                     **kwds):

    # Ref: https://stackoverflow.com/questions/23547347/parallel-coordinates-plot-for-continous-data-in-pandas
    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends = set([])

    ncols = len(df.columns)

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)
        
    fig = plt.figure()
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)),  **kwds)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min,class_max,10)
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    # return fig

def get_eval_metric(y_true, y_pred, y_pred_baseline="auto"):
    error = y_pred - y_true
    abs_error = np.abs(error)
    
    RMeanSE = np.power(np.mean(np.square(error)), 0.5)
    MeanAE = np.mean(abs_error)
    MedianAE = np.median(abs_error)

    if y_pred_baseline == "auto":
        error_baseline = np.mean(y_true) - y_true
    else:
        error_baseline = y_pred_baseline - y_true
    abs_error_baseline = np.abs(error_baseline)
    
    return {
        'RMeanSE': np.round(RMeanSE, 2),
        'MeanAE': np.round(MeanAE, 2),
        'MedianAE': np.round(MedianAE, 2),
        'RMeanSE_r2': np.round((1 - (RMeanSE/(np.power(np.mean(np.square(error_baseline)), 0.5)))), 4),
        'MeanAE_r2': np.round((1 - (MeanAE/(np.mean(abs_error_baseline)))), 4),
        'MedianAE_r2': np.round((1 - (MedianAE/(np.median(abs_error_baseline)))), 4)
    }


def plot_model_importance(rf_model, predictors):
    n_predictors = len(predictors)
    importances = rf_model.feature_importances_
    std = np.std([tree.feature_importances_  for tree in rf_model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(n_predictors), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(n_predictors), list(predictors[i] for i in list(indices)), rotation=45)
    plt.xlim([-1, n_predictors])
    plt.show()
