import csv
import math

import pandas as pd
import plotly.express as px
from tqdm import tqdm


def convert_to_cartesian(lat, long):
    print(f"{lat},{long}")
    phi = math.radians(lat)
    lambda_ = math.radians(long)
    x = math.cos(phi) * math.cos(lambda_)
    y = math.cos(phi) * math.sin(lambda_)
    z = math.sin(phi) + x*.3 - y*.2
    return [x, y, z]


def visualize_3d_topics(items):
    """Reduce to 3D and visualize topics using Plotly."""
    if len(items) > 3000:
        print(f"⚠️ Warning: {len(items)} points may slow down visualization")

    # Plot
    df_plot = pd.DataFrame(items, columns=["x", "y", "z"])

    fig = px.scatter_3d(df_plot, x="x", y="y", z="z",
                        color="z",
                        title="3D Topic Visualization")
    fig.update_traces(marker=dict(size=1))
    fig.write_html("topic_viz_light.html", auto_open=False)
    fig.show()


with open("/Users/matthewstafford/Downloads/simplemaps_worldcities_basicv1.901/worldcities.csv", "r") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(spamreader)
    visualize_3d_topics([convert_to_cartesian(float(row[2]), float(row[3])) for row in tqdm(spamreader)])
