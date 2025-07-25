import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from matplotlib.patches import Patch

LATIN_AMERICAN_COUNTRIES = [
        'Mexico', 'Guatemala', 'Belize', 'Honduras', 'El Salvador', 'Nicaragua', 'Costa Rica', 'Panama',
        'Cuba', 'Jamaica', 'Haiti', 'Dominican Rep.', 'Puerto Rico',  # Caribbean
        'Colombia', 'Venezuela', 'Ecuador', 'Brazil', 'Peru'
    ]
EUROPEAN_COUNTRIES = [
        'France', 'Germany', 'Italy', 'Spain', 'Poland', 'Norway', 'Sweden', 'Finland', 'United Kingdom',
        'Portugal', 'Greece', 'Austria', 'Switzerland', 'Belgium', 'Netherlands', 'Czechia',
        'Slovakia', 'Hungary', 'Denmark', 'Ireland', 'Lithuania', 'Latvia', 'Estonia', 'Croatia',
        'Slovenia', 'Romania', 'Bulgaria', 'Serbia', 'Ukraine', 'Belarus', 'Bosnia and Herz.',
        'North Macedonia', 'Albania', 'Moldova', 'Montenegro', 'Kosovo', 'Luxembourg'
    ]

COLOR_TIME = "#9467BD"  # Purple - Time
COLOR_RMSE = "#5BA698"  # Teal - RMSE
COLOR_R2 = "#D62728"    # Red - R²
COLORS_REGIONS = ["#2CA02C", "#1F77B4", "#9467BD", "#FF7F0E"]
COLORS_DATA = [
    "#2C5F2D",  # deep forest green - all sar
    "#2C5F2D",  # deep forest green - all sar
    "#4682B4",  # steel blue (cooler than sky blue, more defined) - optical
    "#2C5F2D",  # deep forest green - all sar
    "#8FBC8F",  # light green - in sar
    "#4682B4",  # steel blue (cooler than sky blue, more defined) - optical
    "#8FBC8F",  # light green - in sar
    "#4682B4",  # steel blue (cooler than sky blue, more defined) - optical
    "#8FBC8F",  # light green - in sar
    "#4682B4"  # steel blue (cooler than sky blue, more defined) - optical
]

REGION_LABELS = ['Germany', 'Panama', 'Switzerland', 'Italy']

XTICKLABELS = ['50,10', '50,15', '50,30', '125,10', '125,15', '125,30', '200,10', '200,15', '200,30']

def plot_raster(raster, title, cmap, normalized=True, cbar_label=''):
    fig = plt.figure(figsize=(8, 8))

    if normalized:
        vmax = 1
    else:
        vmax = 350

    im = plt.imshow(raster, cmap=cmap, vmin=0,  vmax=vmax)
    cbar = plt.colorbar(im, shrink=0.92, pad=0.02, label=cbar_label)
    cbar.ax.tick_params(labelsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=15)
    plt.show()


def plot_aois_on_map(polygons=['de', 'swiss', 'italy', 'pa']):
    fig, axs = plt.subplots(1, 2, figsize=(10, 20))

    world = gpd.read_file("/Users/annalisefishell/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

    # Panama
    # Filter to only Latin American countries
    latam_raw = world[world['NAME'].isin(LATIN_AMERICAN_COUNTRIES)]
    latam = latam_raw.explode(index_parts=False)
    latam = latam[latam.geometry.centroid.x.between(-120, -30) & latam.geometry.centroid.y.between(-60, 30)]

    polygon_gdf = gpd.GeoDataFrame(geometry=[polygons[-1]], crs="EPSG:4326")

    # Define colors and labels
    colors = ["#a266cc"]
    labels = ['Panama']

    # Plotting
    latam.plot(ax=axs[0], color='#f0f0f0', edgecolor='gray')

    for i, (geom, color) in enumerate(zip(polygon_gdf.geometry, colors)):
        gpd.GeoSeries([geom], crs="EPSG:4326").plot(ax=axs[0], facecolor=color, edgecolor=colors[i], alpha=0.9, label=labels[i])

    # Legend
    legend_elements = [Patch(facecolor=colors[i], edgecolor=colors[i], label=labels[i]) for i in range(len(labels))]
    axs[0].legend(handles=legend_elements, loc='upper right', title='Study region')

    # Clean axes
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_frame_on(False)

    # Title and layout
    axs[0].set_xlim([-105, -65])
    axs[0].set_ylim([-4, 26])
    axs[0].set_title("Latin America", size=18)


    # European countries
    # Filter world map to include only selected European countries
    europe_raw = world[world['NAME'].isin(EUROPEAN_COUNTRIES)]
    europe_exp = europe_raw.explode(index_parts=False)
    europe = europe_exp[europe_exp.geometry.centroid.x > -25]
    europe = europe[europe.geometry.centroid.y > 34]
    europe = europe[europe.geometry.centroid.y < 75]

    polygon_gdf = gpd.GeoDataFrame(geometry=polygons[0:3], crs="EPSG:4326")

    # Define colors and labels
    colors = ["#da4646", "#6699cc", "#17becf"]
    labels = ['Germany', 'Switzerland', 'Italy']

    # Plotting
    europe.plot(ax=axs[1], color='#f0f0f0', edgecolor='gray')

    for i, (geom, color) in enumerate(zip(polygon_gdf.geometry, colors)):
        gpd.GeoSeries([geom], crs="EPSG:4326").plot(ax=axs[1], facecolor=color, edgecolor=colors[i], alpha=0.9, label=labels[i])

    # Legend
    legend_elements = [Patch(facecolor=colors[i], edgecolor=colors[i], label=labels[i]) for i in range(len(labels))]
    axs[1].legend(handles=legend_elements, loc='upper left', title='Study region')

    # Clean axes
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_frame_on(False)

    # Title and layout
    axs[1].set_ylim([34,63])
    axs[1].set_title("Europe", size=18)

    plt.tight_layout()
    plt.show()

def plot_mean_metrics(times, rmses, r2s, ax1, ax2):
    xticklabels = ['50,10', '50,15', '50,30', '125,10', '125,15', '125,30', '200,10', '200,15', '200,30']

    # Plot Running Time
    ax1.plot(times, color=COLOR_TIME, marker='o', linewidth=2)
    ax1.set_title('Running Time', fontsize=16)
    ax1.set_ylabel('Seconds', fontsize=12)
    ax1.set_xlabel('Model parameters (num trees, tree depth)', fontsize=12)
    ax1.set_xticks(range(len(xticklabels)))
    ax1.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot RMSE and R²
    ax2.plot(rmses, color=COLOR_RMSE, marker='s', linewidth=2, label='RMSE')
    axs1 = ax2.twinx()
    axs1.plot(r2s, color=COLOR_R2, marker='^', linewidth=2, label='$R^2$')

    ax2.set_title('RMSE and $R^2$', fontsize=16)
    ax2.set_ylabel('RMSE (Mg/ha)', fontsize=12)
    axs1.set_ylabel('$R^2$', fontsize=12)
    ax2.set_xlabel('Model parameters (num trees, tree depth)', fontsize=12)
    ax2.set_xticks(range(len(xticklabels)))
    ax2.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax2.set_ylim([26.5, 32.5])
    axs1.set_ylim([0.6, 1])
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Combine legends
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = axs1.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=10)

def plot_all_mean_metrics(times, rmses, r2s, ax1, ax2, ax3, i):
    # Plot Running Time
    ax1.plot(times, color=COLORS_REGIONS[i], marker='o', linewidth=2, label=REGION_LABELS[i])
    ax1.set_title('Running Time', fontsize=16)
    ax1.set_ylabel('Seconds', fontsize=12)
    ax1.set_xlabel('Model parameters (num trees, tree depth)', fontsize=12)
    ax1.set_xticks(range(len(XTICKLABELS)))
    ax1.set_xticklabels(XTICKLABELS, rotation=45, ha='right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot RMSE and R²
    ax2.plot(rmses, color=COLORS_REGIONS[i], marker='s', linewidth=2, label=REGION_LABELS[i])
    ax2.set_title('RMSE', fontsize=16)
    ax2.set_ylabel('RMSE (Mg/ha)', fontsize=12)
    ax2.set_xlabel('Model parameters (num trees, tree depth)', fontsize=12)
    ax2.set_xticks(range(len(XTICKLABELS)))
    ax2.set_xticklabels(XTICKLABELS, rotation=45, ha='right')
    ax2.set_ylim([25, 84])
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.plot(r2s, color=COLORS_REGIONS[i], marker='^', linewidth=2, label=REGION_LABELS[i])
    ax3.set_title('$R^2$', fontsize=16)
    ax3.set_ylabel('$R^2$', fontsize=12)
    ax3.set_xlabel('Model parameters (num trees, tree depth)', fontsize=12)
    ax3.set_xticks(range(len(XTICKLABELS)))
    ax3.set_xticklabels(XTICKLABELS, rotation=45, ha='right')
    ax3.set_ylim([0.4, 1])
    ax3.grid(True, linestyle='--', alpha=0.6)

    if i == 3:
        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)
        ax3.legend(fontsize=10)

def plot_input_diff(r2_values, rmse_values, labels, x_positions, gap_centers):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'hspace': 0.5})

    bars = axs[0].bar(x_positions, r2_values, color=COLORS_DATA)

    # Set x-ticks at bar centers with correct labels
    axs[0].set_xticks(x_positions, labels, rotation=45, ha='right', fontsize=10)
    axs[0].set_ylabel('Percentage change', fontsize=12)
    axs[0].set_xlabel('Data type', fontsize=12)
    axs[0].set_title('R²', fontsize=14)

    # Add group labels above bars
    group_centers = [
        np.mean([0, 1, 2]),      # Center of Group 1 (Germany)
        np.mean([4, 5, 6]),      # Center of Group 2 (Panama)
        np.mean([8, 9]),         # Center of Group 3 (Switzerland)
        np.mean([11, 12])        # Center of Group 4 (Italy)
    ]

    for center, label in zip(group_centers, REGION_LABELS):
        axs[0].text(center, max(r2_values) + 0.6, label, ha='center', va='bottom', fontsize=12)

    for edge in gap_centers:
        axs[0].axvline(x=edge, color='gray', linestyle='--', linewidth=1)

    axs[0].set_ylim(0, max(r2_values) + 6.3)  # Add some vertical space for the labels



    bars = axs[1].bar(x_positions, rmse_values, color=COLORS_DATA)

    # Set x-ticks at bar centers with correct labels
    axs[1].set_xticks(x_positions, labels, rotation=45, ha='right', fontsize=10)
    axs[1].set_ylabel('Percentage change', fontsize=12)
    axs[1].set_xlabel('Data type', fontsize=12)
    axs[1].set_title('RMSE', fontsize=14)

    # Add group labels above bars
    group_centers = [
        np.mean([0, 1, 2]),      # Center of Group 1 (Germany)
        np.mean([4, 5, 6]),      # Center of Group 2 (Panama)
        np.mean([8, 9]),         # Center of Group 3 (Switzerland)
        np.mean([11, 12])        # Center of Group 4 (Italy)
    ]

    for center, label in zip(group_centers, REGION_LABELS):
        axs[1].text(center, max(rmse_values) + 0.6, label, ha='center', va='bottom', fontsize=12)

    for edge in gap_centers:
        axs[1].axvline(x=edge, color='gray', linestyle='--', linewidth=1)

    axs[1].set_ylim(0, max(rmse_values) + 13)  # Add some vertical space for the labels

    plt.suptitle('Percentage change of evaluation metrics for different input data \n compared to a model with all inputs used', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.show()