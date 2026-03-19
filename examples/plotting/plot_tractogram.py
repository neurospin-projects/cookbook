"""
Color bundles
=============

Display the bundles using color.

In this example we load a template fiber bundles and want to color each bundle
regarding a specific metric. More specifically we will focus on the GeoLab long,
medium and short templates.

First we import modules and set global parameters (that can be changed).
"""

import requests
import nibabel
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from dipy.viz import actor, window


# Public URL containing a tck file
api_url = (
    "https://api.github.com/repos/vindasna/GeoLab/contents/SGT/sgtTck"
)
out_dir = Path("/tmp/bundles")
out_dir.mkdir(exist_ok=True)
response = requests.get(api_url)
response.raise_for_status()
files = response.json()
for obj in files[:30]:
    if obj["type"] == "file":
        raw_url = obj["download_url"]
        filename = out_dir / obj["name"]
        print(f"Downloading {obj['name']} ...")
        file_data = requests.get(raw_url)
        file_data.raise_for_status()
        filename.write_bytes(file_data.content)
print("Download complete.")


# %%
# Then, we load the TCK data organized per bundles. You can filter the loaded
# data by removing some keys of the dictionary:

bundle_files = out_dir.glob("*.tck")
data = {}
for path in tqdm(bundle_files):
    name = path.stem
    tck = nibabel.streamlines.load(path)
    tractogram = tck.tractogram
    streamlines = tractogram.streamlines
    data[name] = streamlines


# %%
# Then, render bundles and metrics on bundles. Here we generate random metrics
# and use the plasma color map:

rng = np.random.default_rng()
cmap = mpl.colormaps["plasma"]
my_metric = [cmap(val)[:3] for val in rng.random(len(data))]
scene = window.Scene()
for idx, (name, bundle) in enumerate(data.items()):
    stream_actor = actor.line(bundle, colors=my_metric[idx], linewidth=0.1)
    scene.add(stream_actor)
scene.set_camera(
    position=(-351.46, -31.52, 70.51),
    focal_point=(-0.29, -19.66, 7.92),
    view_up=(0.18, 0.00, 0.98)
)
# window.show(scene, size=(600, 600), reset_camera=True)
scene.camera_info()
window.record(
    scene=scene,
    out_path=out_dir / "bundles1.png",
    size=(600, 600),
)
scene.set_camera(
    position=(514.35, 64.79, -24.62),
    focal_point=(-0.29, -19.66, 7.92),
    view_up=(0.06, 0.00, 1.00)
)
window.record(
    scene=scene,
    out_path=out_dir / "bundles2.png",
    size=(600, 600),
)
scene.set_camera(
    position=(-14.07, 31.80, 527.73),
    focal_point=(-0.29, -19.66, 7.92),
    view_up=(0.06, -0.99, 0.10)
)
window.record(
    scene=scene,
    out_path=out_dir / "bundles3.png",
    size=(600, 600),
)


# %%
# Finally, we concatenate images horizontally:

images = [
    Image.open(out_dir / f"{name}.png")
    for name in ["bundles2", "bundles3", "bundles1"]
]
images = [im.crop((100, 100, 500, 500)) for im in images]
widths, heights = zip(*(im.size for im in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new("RGB", (total_width, max_height))
x_offset = 0
for im in images:
  new_im.paste(im, (x_offset, 0))
  x_offset += im.size[0]
new_im.save(out_dir / "bundles.png")
plt.imshow(new_im)
ax = plt.gca()
ax.axis("off")

