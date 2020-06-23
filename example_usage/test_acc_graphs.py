import matplotlib.pyplot as plt
import numpy as np


def print_test_acc(test_accs):
    mean = []
    std = []
    x = np.array(range(10, 101, 10))

    for i, _ in enumerate(x):
        column = np.array([row[i] for row in test_accs])
        mean.append(np.mean(column))
        std.append(np.std(column))

    y = np.array(mean)
    e = np.array(std)

    fig, ax = plt.subplots()
    fontsize=15

    ax.errorbar(x, y, e, marker='.', capsize=3)
    ax.yaxis.grid(True)
    plt.xticks(x, fontsize=fontsize)
    plt.xlabel('Number of classes', fontsize=fontsize)

    plt.yticks(x / 100.0, fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.show()


if __name__ == "__main__":
    cf1 = [0.9206, 0.4925, 0.3319, 0.2483, 0.1994, 0.1656, 0.1423, 0.1243, 0.1109, 0.09982]

    cf2 = [0.8722, 0.4911, 0.3317, 0.24905, 0.19936, 0.166, 0.1425, 0.1242, 0.1107, 0.0998]

    cf3 = [0.837, 0.4827, 0.3319, 0.2491, 0.1994, 0.1664, 0.14257, 0.124275, 0.11076, 0.09982]

    cf = [cf1, cf2, cf3]
    print_test_acc(cf)

    lwf1 = [0.9096, 0.7014, 0.5729, 0.451, 0.3502, 0.2996, 0.2592, 0.2025, 0.181, 0.1567]

    lwf2 = [0.8246, 0.6682, 0.5598, 0.4503, 0.37312, 0.3129, 0.2967, 0.2411, 0.1995, 0.17938]

    lwf3 = [0.8334, 0.6698, 0.5765, 0.4544, 0.36008, 0.306, 0.28046, 0.23875, 0.20393, 0.18594]

    lwf = [lwf1, lwf2, lwf3]
    print_test_acc(lwf)

    icarl1 = [0.9174, 0.8556, 0.7743, 0.6937, 0.6354, 0.5865, 0.5486, 0.511, 0.4849, 0.4514]

    icarl2 = [0.9132, 0.8437, 0.74133, 0.6573, 0.59616, 0.55347, 0.514057, 0.47695, 0.45549, 0.4229]

    icarl3 = [0.894, 0.8581, 0.781, 0.71025, 0.64536, 0.59693, 0.5583, 0.51785, 0.49747, 0.46238]

    icarl = [icarl1, icarl2, icarl3]
    print_test_acc(icarl)

