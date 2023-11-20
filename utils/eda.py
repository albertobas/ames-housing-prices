import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_univariate(X, value_vars, font_scale, col_wrap, height, aspect, function, bins=None, labels_thresh=None, rotation=None):
    sns.set(font_scale=font_scale, style='darkgrid')
    if function == sns.barplot:
        plot_data = pd.DataFrame(X[value_vars].unstack()).reset_index()
        plot_data.columns = ['predictor', 'count', 'predictor_value']
        plot_data = plot_data.groupby(['predictor', 'predictor_value']).count()
        plot_data['predictor'] = plot_data.index.get_level_values(0)
        if (X.dtypes[value_vars] == 'object').any():
            plot_data['predictor_value'] = plot_data.index.get_level_values(1)
            plot_data = plot_data.sort_values(ascending=False, by="count")
        else:
            plot_data['predictor_value'] = plot_data.index.get_level_values(
                1).astype('int64')
        fgmap = sns.FacetGrid(plot_data, col="predictor", col_wrap=col_wrap, height=height, aspect=aspect, sharex=False,
                              sharey=True, col_order=value_vars).map(function, 'predictor_value', 'count', color='#5876a0')
        if rotation is not None:
            if labels_thresh is None:
                [plt.setp(ax.get_xticklabels(), rotation=rotation)
                 for ax in fgmap.axes.flat]
            else:
                [plt.setp(ax.get_xticklabels(), rotation=rotation) for ax in fgmap.axes.flat if
                 len(ax.get_xticklabels()) > labels_thresh]
    else:
        plot_data = pd.melt(X, value_vars=value_vars,
                            var_name="predictor", value_name="predictor_value")
        if function == sns.boxplot:
            fgmap = sns.FacetGrid(plot_data, col="predictor", col_wrap=col_wrap, height=height, aspect=aspect, sharex=False,
                                  sharey=False).map(function, "predictor_value", orient="v", width=0.2)
        elif function == sns.distplot:
            sns.FacetGrid(plot_data, col="predictor", col_wrap=col_wrap, height=height, aspect=aspect, sharex=False,
                          sharey=False).map(function, "predictor_value", bins=bins)


def plot_bivariate(X, id_vars, value_vars, font_scale, col_wrap, height, aspect, function, labels_thresh=None, rotation=None):
    sns.set(font_scale=font_scale, style='darkgrid')
    plot_data = pd.melt(X, id_vars=id_vars, value_vars=value_vars,
                        var_name="predictor", value_name="predictor_value")
    fgmap = sns.FacetGrid(plot_data, col="predictor", col_wrap=col_wrap, height=height, aspect=aspect, sharex=False,
                          sharey=True).map(function, "predictor_value", id_vars)
    if rotation is not None:
        if labels_thresh is None:
            [plt.setp(ax.get_xticklabels(), rotation=rotation)
             for ax in fgmap.axes.flat]
        else:
            [plt.setp(ax.get_xticklabels(), rotation=rotation) for ax in fgmap.axes.flat if
             len(ax.get_xticklabels()) > labels_thresh]
