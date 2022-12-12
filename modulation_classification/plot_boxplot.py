import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
#import tikzplotlib
#import numpy as np

def aggregate_dict_to_pd(df, path,annotation):
    if df is None:
        with open(path, 'rb') as f:
            dict_for_pd =   pickle.load(f)
        df =                pd.DataFrame(dict_for_pd)
        df['details'] =     annotation
    else:
        with open(path, 'rb') as f:
            new_dict =      pickle.load(f)
        new_df =            pd.DataFrame(new_dict)
        new_df['details'] = annotation
        #df =                df.append(new_df, ignore_index=True, sort=False) # use concat instead! append will be deprecated
        df =                pd.concat([df,new_df])
    return df

def color_the_boxplot(bp):
    colors = ['m', 'royalblue', 'red', 'chocolate', 'mediumseagreen', 'royalblue', 'k', 'burlywood',
              'mediumaquamarine', 'rosybrown', 'forestgreen', 'orangered', 'teal', 'mediumorchid', 'skyblue', 'salmon',
              'lightblue', 'darkcyan', 'thistle', 'red', 'mediumseagreen', 'indianred', 'navy', 'mediumturquoise',
              'saddlebrown', 'darkslategrey', 'slategray', 'rosybrown','black']
    #for row_key, (ax,row) in bp.iteritems():
    for row_key, (ax, row) in bp.items():
        ax.set_xlabel('')
        ax.set_ylabel(ax.get_title())
        ax.set_title('')
        for i,tick in enumerate(ax.get_xticklabels()):
            tick.set_rotation(90)
            tick.set_color(colors[i])
        for i,box in enumerate(row['boxes']):
            box.set_color(colors[i])
            box.set_linewidth(1.5)
        for i,whisker in enumerate(row['whiskers']):
            whisker.set_color(colors[i//2])
            whisker.set_linewidth(1.5)
        for i,caps in enumerate(row['caps']):
            caps.set_color(colors[i//2])
            caps.set_linewidth(1.5)
        for i,fliers in enumerate(row['fliers']):
            fliers.set_color(colors[i])
            fliers.set_linewidth(1.5)
            fliers.set_mfc(colors[i])
            fliers.set_mec(colors[i])
            fliers.set_markersize(1.5)
        for i,medians in enumerate(row['medians']):
            medians.set_color(colors[i])
            medians.set_linewidth(1.5)
        for i,means in enumerate(row['means']):
            means.set_color(colors[i])
            means.set_linewidth(2)
            means.set_linestyle('--')
    plt.suptitle('')
    plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
    plt.gca()

run_folder_str =        "run_2022_11_17__14_40"

list_boxes =        [   run_folder_str + "/saved/alpha_0.1N_3600set_predictor_NPBlr_inner_0.02inner_iter4000dataset_for_box_plot",
                        run_folder_str + "/saved/alpha_0.1N_3600set_predictor_VBlr_inner_0.02inner_iter4000dataset_for_box_plot",
                        run_folder_str + "/saved/alpha_0.1N_3600set_predictor_CV+6lr_inner_0.02inner_iter4000dataset_for_box_plot",
                        run_folder_str + "/saved/alpha_0.1N_3600set_predictor_CV+12lr_inner_0.02inner_iter4000dataset_for_box_plot" ]
list_annotation =   [   "1NPB", # the first char is for ordering and should be removed manually
                        "2VB",
                        "36-CV",
                        "412-CV",]
matplotlib.use('Qt5Agg')
df =                        None
for curr_path,annotation in zip(list_boxes,list_annotation):
    df =                    aggregate_dict_to_pd(   df,
                                                    curr_path,
                                                    annotation)
boxplots_list =             ['Inefficiency','Coverage']#,'Accuracy tr','Accuracy te']
for boxplot_str in boxplots_list:
    bp =                    df.boxplot(             boxplot_str,
                                                    by="details",
                                                    return_type='both',
                                                    patch_artist = False,
                                                    widths=0.75,
                                                    figsize=(4,4),
                                                    showmeans=True,
                                                    meanline=True)
    color_the_boxplot(bp)
    plt.savefig(boxplot_str + '.png', dpi=350)
plt.show() # only once, at the end



#for boxplot_str in boxplots_list:
#    plt.savefig(boxplot_str + '.png', dpi=350)
#    #tikzplotlib.get_tikz_code('fig_tikz_'+ boxplot_str + '.tex')
