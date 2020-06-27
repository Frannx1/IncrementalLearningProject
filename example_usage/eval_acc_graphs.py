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


def print_test_acc_means_together(test_accs, legends, log=False):
    means, stds = [], []
    x = np.array(range(10, 101, 10))

    for acc in test_accs:
        mean, std = [], []
        for i, _ in enumerate(x):
            column = np.array([row[i] for row in acc])
            mean.append(np.mean(column))
            std.append(np.std(column))
        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots()
    fontsize = 15
    for y, e in zip(np.array(means), np.array(stds)):
        ax.errorbar(x, y, e, marker='.', capsize=3)

    ax.yaxis.grid(True)
    if log:
        ax.set_yscale('log')

    plt.legend(legends)
    plt.xticks(x, fontsize=fontsize)
    plt.xlabel('Number of classes', fontsize=fontsize)

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
    #print_test_acc_means(cf)
    print('Mean cf: {}'.format(np.array(cf).mean()))

    lwf1 = [0.9096, 0.7014, 0.5729, 0.451, 0.3502, 0.2996, 0.2592, 0.2025, 0.181, 0.1567]

    lwf2 = [0.8246, 0.6682, 0.5598, 0.4503, 0.37312, 0.3129, 0.2967, 0.2411, 0.1995, 0.17938]

    lwf3 = [0.8334, 0.6698, 0.5765, 0.4544, 0.36008, 0.306, 0.28046, 0.23875, 0.20393, 0.18594]

    lwf = [lwf1, lwf2, lwf3]
    #print_test_acc_means(lwf)
    print('Mean lwf: {}'.format(np.array(lwf).mean()))

    icarl1 = [0.9174, 0.8556, 0.7743, 0.6937, 0.6354, 0.5865, 0.5486, 0.511, 0.4849, 0.4514]

    icarl2 = [0.9132, 0.8437, 0.74133, 0.6573, 0.59616, 0.55347, 0.514057, 0.47695, 0.45549, 0.4229]

    icarl3 = [0.894, 0.8581, 0.781, 0.71025, 0.64536, 0.59693, 0.5583, 0.51785, 0.49747, 0.46238]

    icarl = [icarl1, icarl2, icarl3]
    #print_test_acc_means(icarl)
    print('Mean icarl: {}'.format(np.array(icarl).mean()))

    print_test_acc_means_together([cf, lwf, icarl], ['Fine tunning', 'LwF', 'iCaRL'])

    icarl_hybrid1 = [0.8873, 0.8427, 0.7792, 0.6804, 0.6211, 0.5414, 0.5106, 0.4521, 0.4043, 0.3695]

    icarl_hybrid2 = [0.9214, 0.8542, 0.7713, 0.6844, 0.6322, 0.5533, 0.5072, 0.4696, 0.4191, 0.3756]

    icarl_hybrid3 = [0.9388, 0.8467, 0.7436, 0.6621, 0.5978, 0.5342, 0.4838, 0.4339, 0.3911, 0.3403]

    icarl_hybrid = [icarl_hybrid1, icarl_hybrid2, icarl_hybrid3]
    #print_test_acc_means(icarl_hybrid)
    print('Mean icarl_hybrid: {}'.format(np.array(icarl_hybrid).mean()))

    #print_test_acc_means_together([icarl, icarl_hybrid], ['NME', 'FCC'])

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
    #print_test_acc_together(lwf_ex, legends)

    icarl_bce_bce = [0.9206, 0.8644, 0.7857, 0.707, 0.6364, 0.5917, 0.5535, 0.5112, 0.4919, 0.4569]

    icarl_bce_l2 = [0.8752, 0.7053, 0.5585, 0.4726, 0.4077, 0.3582, 0.3306, 0.302, 0.2798, 0.2598]

    icarl_ce_bce = [0.859, 0.7991, 0.7185, 0.6458, 0.5823, 0.5357, 0.4977, 0.4574, 0.4281, 0.3906]

    icarl_ce_ce = [0.943, 0.7672, 0.604, 0.5023, 0.4389, 0.3916, 0.3498, 0.3062, 0.277, 0.2457]

    icarl_ce_l2 = [0.5934, 0.5348, 0.4723, 0.3959, 0.3332, 0.2915, 0.2701, 0.2118, 0.2007, 0.1705]

    icarl_ex = [icarl_bce_bce, icarl_bce_l2, icarl_ce_bce, icarl_ce_ce, icarl_ce_l2]
    #print_test_acc_together(icarl_ex, legends)


    ############################################
    # Now the ones with different clasifier
    ############################################

    icarl_knn_10_1 = [0.917, 0.8551, 0.7559, 0.6673, 0.6058, 0.5535, 0.5191, 0.4779, 0.4466, 0.4177]

    icarl_knn_10_2 = [0.9026, 0.844, 0.7501, 0.6676, 0.6006, 0.5506, 0.514, 0.4742, 0.4463, 0.4148]

    icarl_knn_10_3 = [0.8911, 0.8355, 0.7527, 0.6524, 0.6110, 0.5565, 0.5183, 0.4781, 0.4397, 0.4138]

    icarl_knn_10 = [icarl_knn_10_1, icarl_knn_10_2, icarl_knn_10_2]
    print('Mean knn_10: {}'.format(np.array(icarl_knn_10).mean()))

    icarl_knn_3_1 = [0.8382, 0.8164, 0.7383, 0.6536, 0.5804, 0.5368, 0.4968, 0.4502, 0.4234, 0.3994]

    icarl_knn_3_2 = [0.919, 0.8442, 0.7459, 0.6501, 0.5743, 0.5148, 0.4795, 0.4347, 0.4101, 0.3806]

    icarl_knn_3_3 = [0.8902, 0.8259, 0.7521, 0.6682, 0.5927, 0.5356, 0.4924, 0.448, 0.4202, 0.385]

    icarl_knn_3 = [icarl_knn_3_1, icarl_knn_3_2, icarl_knn_3_2]
    print('Mean knn_3: {}'.format(np.array(icarl_knn_3).mean()))

    icarl_knn_5_1 = [0.8886, 0.8344, 0.7573, 0.661, 0.5955, 0.5381, 0.5016, 0.4558, 0.4315, 0.4017]

    icarl_knn_5_2 = [0.88, 0.8356, 0.7637, 0.6935, 0.6303, 0.5746, 0.5343, 0.4891, 0.4632, 0.4307]

    icarl_knn_5_3 = [0.92, 0.8717, 0.7834, 0.7042, 0.6378, 0.5849, 0.545, 0.4966, 0.4749, 0.4378]

    icarl_knn_5 = [icarl_knn_5_1, icarl_knn_5_2, icarl_knn_5_2]
    print('Mean knn_5: {}'.format(np.array(icarl_knn_5).mean()))

    #print_test_acc_means(icarl_knn_10)
    #print_test_acc_means_together([icarl_knn_10, icarl_knn_3, icarl_knn_5], ['KNN-10', 'KNN-3', 'KNN-5'])

    #print_test_acc_means_together([icarl, icarl_knn_5], ['NME', 'KNN-5'])

    icarl_svc_default_1 = [0.8808, 0.8545, 0.7767, 0.7071, 0.651, 0.6092, 0.5674, 0.5189, 0.4958, 0.4538]

    icarl_svc_default_2 = [0.9094, 0.8369, 0.7639, 0.6878, 0.6277, 0.5732, 0.5379, 0.4951, 0.4765, 0.438]

    icarl_svc_default_3 = [0.9012, 0.8388, 0.7605, 0.6824, 0.6289, 0.5766, 0.5417, 0.5053, 0.4878, 0.441]
    icarl_svc_default = [icarl_svc_default_1, icarl_svc_default_2, icarl_svc_default_3]
    print('Mean svc 1: {}'.format(np.array(icarl_svc_default).mean()))

    icarl_svc_01_1 = [0.865, 0.7905, 0.7071, 0.6198, 0.5597, 0.5145, 0.4754, 0.4387, 0.4194, 0.3846]

    icarl_svc_01_2 = [0.9214, 0.8474, 0.7539, 0.6671, 0.5919, 0.5434, 0.5092, 0.4688, 0.4428, 0.4082]

    icarl_svc_01_3 = [0.7608, 0.7848, 0.7265, 0.6556, 0.5997, 0.5442, 0.5059, 0.4668, 0.4466, 0.4148]
    icarl_svc_01 = [icarl_svc_01_1, icarl_svc_01_2, icarl_svc_01_3]
    print('Mean svc 0.1: {}'.format(np.array(icarl_svc_01).mean()))

    icarl_svc_10_1 = [0.8906, 0.8219, 0.7384, 0.6549, 0.5879, 0.5469, 0.5109, 0.4672, 0.4475, 0.4108]

    icarl_svc_10_2 = [0.8242, 0.8193, 0.7456, 0.6743, 0.6142, 0.5665, 0.5345, 0.4895, 0.4644, 0.431]

    icarl_svc_10_3 = [0.7976, 0.781, 0.7183, 0.6415, 0.5799, 0.5304, 0.5039, 0.4706, 0.4486, 0.4138]
    icarl_svc_10 = [icarl_svc_10_1, icarl_svc_10_2, icarl_svc_10_3]
    print('Mean svc 10: {}'.format(np.array(icarl_svc_10).mean()))

    #print_test_acc_means(icarl_svc_default)
    print_test_acc_means_together([icarl_svc_01, icarl_svc_default, icarl_svc_10], ['SVC C = 0.1', 'SVC C = 1', 'SVC C = 10'])
    print_test_acc_means_together([icarl, icarl_svc_default], ['NME', 'SVC C = 1'])

    icarl_cos_1 = [0.945, 0.6641, 0.5247, 0.427, 0.3913, 0.3454, 0.3207, 0.2927, 0.2757, 0.2603]

    icarl_cos_2 = [0.8704, 0.57, 0.4114, 0.3198, 0.2676, 0.2358, 0.2117, 0.1893, 0.1723, 0.1579]

    icarl_cos_3 = [0.8994, 0.6315, 0.4784, 0.3977, 0.3532, 0.2954, 0.2618, 0.2434, 0.2353, 0.2197]

    icarl_cos = [icarl_cos_1, icarl_cos_2, icarl_cos_3]
    #print_test_acc_means(icarl_cos)
    print_test_acc_means_together([icarl, icarl_cos], ['NME', 'COS (BCE + COS)'])

    print_test_acc_means_together([icarl, icarl_cos, icarl_svc_default, icarl_knn_10], ['NME', 'COS', 'SVC', 'KNN'])

    cos_bce_l2_1 = [0.8806, 0.6963, 0.5455, 0.4553, 0.3834, 0.3344, 0.3099, 0.2653, 0.2441, 0.2303]

    cos_bce_l2_2 = [0.887, 0.6979, 0.5272, 0.4247, 0.3654, 0.3142, 0.2834, 0.2536, 0.2383, 0.2167]

    cos_bce_l2_3 = [0.8822, 0.6894, 0.5287, 0.4318, 0.3694, 0.3187, 0.2919, 0.2577, 0.2364, 0.2214]

    icarl_cos = [cos_bce_l2_1, cos_bce_l2_2, cos_bce_l2_3]
    #print_test_acc_means(icarl_cos)
    print_test_acc_means_together([icarl, icarl_cos], ['NME', 'COS (BCE + L2)'])
