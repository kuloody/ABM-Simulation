import pandas as pd
import jenkspy
import matplotlib.pyplot as plt
import numpy as np

#class DataBreak(data):

def TwoBreaks(data):


            #data.sort_values(by='E_maths')
            #data = data.reset_index()  # make sure indexes pair with number of rows

            #df['quantile'] = pd.qcut(df['e_maths'], q=3, labels=['group_1', 'group_2','group_3'])
            breaks = jenkspy.jenks_breaks(data['E_math'], nb_class=2)
            data['Classification'] = pd.cut(data['E_math'],
                        bins=breaks,
                        labels=[1, 2],
                        include_lowest=True)

            Group1 = data[data ['Classification'] == 1]
            Group2 = data[data ['Classification'] == 2]

            dataConvert = data.to_numpy()
            Group11 = Group1.to_numpy()
            Group22 = Group2.to_numpy()

            print(data)
            FGroup = Group11[:,8]
            SGroup = Group22[:,8]


            # Create a figure instance
            fig = plt.figure(1, figsize=(9, 6))

            # Create an axes instance
            ax = fig.add_subplot(111)

            # Create the boxplot
             ## add patch_artist=True option to ax.boxplot()
            ## to get fill color
            bp = ax.boxplot((FGroup, SGroup), patch_artist=True)
            ## change outline color, fill color and linewidth of the boxes
            for box in bp['boxes']:
             # change outline color
              box.set( color='#7570b3', linewidth=2)
              # change fill color
              box.set( facecolor = '#1b8f9e' )

            ## change color and linewidth of the whiskers
            for whisker in bp['whiskers']:
             whisker.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the caps
            for cap in bp['caps']:
              cap.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the medians
            for median in bp['medians']:
             median.set(color='#b2df8a', linewidth=2)

            ## change the style of fliers and their fill
            for flier in bp['fliers']:
              flier.set(marker='o', color='#e7298a', alpha=0.5)
            ax.set_xticklabels(['Group1', 'Group2', 'Group3'])

            ax.set_ylabel('End Math')
            plt.show(bp)
            print("Classes are", Group11)
            dataConvert = data.to_numpy()
            plt.scatter(dataConvert[:,4], dataConvert[:,8],  s=50, alpha=0.5)
            print("Y is",dataConvert[:,8])
            plt.ylabel('Emath Score')
            plt.xlabel('Inattintiveness')
            plt.show()

def ThreeBreaks(data):
            breaks = jenkspy.jenks_breaks(data['E_math'], nb_class=3)
            data['Classification'] = pd.cut(data['E_math'],
                        bins=breaks,
                        labels=[1, 2, 3],
                        include_lowest=True)

            Group1 = data[data ['Classification'] == 1]
            Group2 = data[data ['Classification'] == 2]
            Group3 = data[data ['Classification'] == 3]
            dataConvert = data.to_numpy()
            Group11 = Group1.to_numpy()
            Group22 = Group2.to_numpy()
            Group33 = Group3.to_numpy()
            print(data)
            FGroup = Group11[:,8]
            SGroup = Group22[:,8]
            TGroup = Group33[:,8]

            # Create a figure instance
            fig = plt.figure(1, figsize=(9, 6))

            # Create an axes instance
            ax = fig.add_subplot(111)

            # Create the boxplot
             ## add patch_artist=True option to ax.boxplot()
            ## to get fill color
            bp = ax.boxplot((FGroup, SGroup, TGroup), patch_artist=True)
            ## change outline color, fill color and linewidth of the boxes
            for box in bp['boxes']:
             # change outline color
              box.set( color='#7570b3', linewidth=2)
              # change fill color
              box.set( facecolor = '#1b8f9e' )

            ## change color and linewidth of the whiskers
            for whisker in bp['whiskers']:
             whisker.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the caps
            for cap in bp['caps']:
              cap.set(color='#7570b3', linewidth=2)

            ## change color and linewidth of the medians
            for median in bp['medians']:
             median.set(color='#b2df8a', linewidth=2)

            ## change the style of fliers and their fill
            for flier in bp['fliers']:
              flier.set(marker='o', color='#e7298a', alpha=0.5)
            ax.set_xticklabels(['Group1', 'Group2', 'Group3'])

            ax.set_ylabel('End Math')
            plt.show(bp)
            print("Classes are", Group11)
            dataConvert = data.to_numpy()
            plt.scatter(dataConvert[:,4], dataConvert[:,8],  s=50, alpha=0.5)
            print("Y is",dataConvert[:,8])
            plt.ylabel('Emath Score')
            plt.xlabel('Inattintiveness')
            plt.show()
