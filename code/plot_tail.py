import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_tail(alpha, accuracy, model_name='default'): 
    data = {'Tail size': alpha, model_name: accuracy}
    df = pd.DataFrame(data, dtype=float)
    df = pd.melt(df, ['Tail size'], var_name="Models", value_name="Accuracy")
    ax = sns.lineplot(x="Tail size", y="Accuracy", hue="Models", style="Models", data=df, markers=False, ci=None)
    plt.xscale('log')
    plt.ylim(0, 100)
    plt.savefig('plot/tail_plot_%s.pdf'%model_name)
    plt.close()
