import matplotlib.pyplot as plt
import geopandas as gpd
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


def plot_aois_on_map(polygons=['de', 'swiss', 'italy', 'pa']):
    fig, axs = plt.subplots(1, 2, figsize=(10, 20))

    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

    # Panama
    # Filter to only Latin American countries
    latam_raw = world[world['NAME'].isin(LATIN_AMERICAN_COUNTRIES)]
    latam = latam_raw.explode(index_parts=False)
    latam = latam[latam.geometry.centroid.x.between(-120, -30) & latam.geometry.centroid.y.between(-60, 30)]

    polygon_gdf = gpd.GeoDataFrame(geometry=polygons[-1], crs="EPSG:4326")

    # Define colors and labels
    colors = ['indigo']
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
    colors = ['green', 'red', 'blue']
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
