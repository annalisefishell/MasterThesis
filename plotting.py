import matplotlib.pyplot as plt

def plot_raster(raster, title, cmap, normalized=True, cbar_label=''):
    fig = plt.figure(figsize=(8, 8))

    if normalized:
        vmax = 1
    else:
        vmax = 350

    im = plt.imshow(raster, cmap=cmap, vmin=0,  vmax=vmax)
    cbar = plt.colorbar(im, shrink=0.8, pad=0.07, label=cbar_label)
    cbar.ax.tick_params(labelsize=12)
    plt.title(title)
    plt.show()