def create_ch9_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    ch9_vmin = 200.
    ch9_vmax = 320.
    ch9_vm1 = 227.
    jet_cnt = int(512 * (ch9_vm1 - ch9_vmin) / (ch9_vmax - ch9_vmin))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([jetcolors[::-1], graycolors], axis=0)
    ch9_cm = ListedColormap(newcolors)
    return ch9_cm



def create_btd_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    btd_vmin = -80
    btd_vmax = 3.3
    btd_vm1 = 0.
    jet_cnt = int(512 * (btd_vmax - btd_vm1) / (btd_vmax - btd_vmin))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([graycolors[::-1], jetcolors], axis=0)
    btd_cm = ListedColormap(newcolors)
    return btd_cm


def create_ch5_cmap():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    ch5_vmin = 200.
    ch5_vmax = 320.
    ch5_vm1 = 223.
    jet_cnt = int(512 * (ch5_vm1 - ch5_vmin) / (ch5_vmax - ch5_vmin))
    gray_cnt = 512 - jet_cnt
    jet = cm.get_cmap('jet', jet_cnt)
    gray = cm.get_cmap('gray', gray_cnt)
    jetcolors = jet(np.linspace(0, 1, jet_cnt))
    graycolors = gray(np.linspace(0.4, 1, gray_cnt))
    newcolors = np.concatenate([jetcolors[::-1], graycolors[::-1]], axis=0)
    newcm = ListedColormap(newcolors)
    return newcm