import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
def print_pie(Accuracy):
    # Data to plot
    labels = 'True pred', 'False pred'
    sizes = [Accuracy, 1 - Accuracy]
    colors = ['yellowgreen', 'lightcoral']
    explode = (0, 0.05)  # explode 2st slice
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
     
    plt.axis('equal')
    plt.show()

def print_bars(performance):
    objects = ('business    ', 'entertainment', 'health', 'politics', 'sports', 'technology')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, performance, align='center', alpha=0.9, width=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('probability')
    plt.title('Classes probabilities')
    plt.show()
    
def print_bars_for_acc(avg_accuracies):
    objects = ('0.001','0.01','0.1','1','10','100', '150', '300')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, avg_accuracies, align='center', alpha=0.9, width=0.5, color = 'orange')
    plt.xticks(y_pos, objects)
    plt.ylabel('Average of accuracies')
    plt.xlabel('C value')
    plt.title('The change of the accuracies w.r.t change in C value')
    plt.show()