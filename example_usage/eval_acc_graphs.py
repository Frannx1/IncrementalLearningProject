import matplotlib.pyplot as plt
import numpy as np


def print_test_acc_means(test_accs):
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
    fontsize = 15

    ax.errorbar(x, y, e, marker='.', capsize=3)
    ax.yaxis.grid(True)
    plt.xticks(x, fontsize=fontsize)
    plt.xlabel('Number of classes', fontsize=fontsize)

    plt.yticks(x / 100.0, fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.show()


def print_test_acc_together(test_accs, legends):
    x = np.array(range(10, 101, 10))

    fig, ax = plt.subplots()
    fontsize = 15

    for acc in test_accs:
        plt.plot(x, acc, marker='.')

    ax.yaxis.grid(True)
    plt.legend(legends)
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
    print_test_acc_means(cf)

    lwf1 = [0.9096, 0.7014, 0.5729, 0.451, 0.3502, 0.2996, 0.2592, 0.2025, 0.181, 0.1567]

    lwf2 = [0.8246, 0.6682, 0.5598, 0.4503, 0.37312, 0.3129, 0.2967, 0.2411, 0.1995, 0.17938]

    lwf3 = [0.8334, 0.6698, 0.5765, 0.4544, 0.36008, 0.306, 0.28046, 0.23875, 0.20393, 0.18594]

    lwf = [lwf1, lwf2, lwf3]
    print_test_acc_means(lwf)

    icarl1 = [0.9174, 0.8556, 0.7743, 0.6937, 0.6354, 0.5865, 0.5486, 0.511, 0.4849, 0.4514]

    icarl2 = [0.9132, 0.8437, 0.74133, 0.6573, 0.59616, 0.55347, 0.514057, 0.47695, 0.45549, 0.4229]

    icarl3 = [0.894, 0.8581, 0.781, 0.71025, 0.64536, 0.59693, 0.5583, 0.51785, 0.49747, 0.46238]

    icarl = [icarl1, icarl2, icarl3]
    print_test_acc_means(icarl)

    ############################################
    # Now the ones with different loss functions
    ############################################
    legends = ['BCE + BCE', 'BCE + L2', 'CE + BCE', 'CE + CE', 'CE + L2']

    lwf_bce_bce = [0.9096, 0.7014, 0.5729, 0.451, 0.3502, 0.2996, 0.2592, 0.2025, 0.181, 0.1567]

    lwf_bce_l2 = [0.8110, 0.6367, 0.4953, 0.4163, 0.3436, 0.3044, 0.2794, 0.236, 0.2246, 0.2069]

    lwf_ce_bce = [0.8156, 0.4839, 0.3271, 0.2458, 0.1972, 0.1633, 0.1404, 0.1219, 0.11, 0.0993]

    lwf_ce_ce = [0.9796, 0.7421, 0.5160, 0.3559, 0.2771, 0.2144, 0.1716, 0.1403, 0.1109, 0.0781]

    lwf_ce_l2 = [0.7978, 0.4767, 0.3955, 0.3122, 0.2689, 0.231, 0.212, 0.1726, 0.1614, 0.1432]

    lwf_ex = [lwf_bce_bce, lwf_bce_l2, lwf_ce_bce, lwf_ce_ce, lwf_ce_l2]
    print_test_acc_together(lwf_ex, legends)

    icarl_bce_bce = [0.9206, 0.8644, 0.7857, 0.707, 0.6364, 0.5917, 0.5535, 0.5112, 0.4919, 0.4569]

    icarl_bce_l2 = [0.8752, 0.7053, 0.5585, 0.4726, 0.4077, 0.3582, 0.3306, 0.302, 0.2798, 0.2598]

    icarl_ce_bce = [0.859, 0.7991, 0.7185, 0.6458, 0.5823, 0.5357, 0.4977, 0.4574, 0.4281, 0.3906]

    icarl_ce_ce = [0.943, 0.7672, 0.604, 0.5023, 0.4389, 0.3916, 0.3498, 0.3062, 0.277, 0.2457]

    icarl_ce_l2 = [0.5934, 0.5348, 0.4723, 0.3959, 0.3332, 0.2915, 0.2701, 0.2118, 0.2007, 0.1705]

    icarl_ex = [icarl_bce_bce, icarl_bce_l2, icarl_ce_bce, icarl_ce_ce, icarl_ce_l2]
    print_test_acc_together(icarl_ex, legends)


