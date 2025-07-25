import os
import json
import matplotlib.pyplot as plt

def visualize(res_ls, info = "max_accu", linewidth = 2, title = "Plot",
              ylabel = "", xlabel = "epoch", step = 1):
    """
    Params:
        res_ls: list of tuple, each tuple have 2 elements, first one is name of the line,
                second one is the json path of that line.
        info:   train_loss, test_loss or max_accu
        step:   plot point after step, like if ar=[2,5,7,9] and step=2, we'll only plot [5,9].
    """
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    for ele in res_ls:
        if os.path.exists(ele[1]):
            # pylint: disable=unspecified-encoding
            with open(ele[1]) as json_data:
                a = json.load(json_data)
                json_data.close()

            a = a[1:]
            plot_list = []
            # pylint: disable=consider-using-enumerate
            for i in range(len(a)):
                if (i+1)%step == 0:
                    m = -1 if info != 'train_loss' else 1e9
                    for j in range(i+1-step,i+1):
                        m = max(m, a[j][info]) if info != 'train_loss' else min(m, a[j][info])
                    plot_list.append(m)

            plt.plot(plot_list, linewidth=linewidth)
    plt.legend([ele[0] for ele in res_ls if os.path.exists(ele[1])])
    plt.show()
