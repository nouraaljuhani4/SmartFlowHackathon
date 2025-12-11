import folium
from folium.plugins import HeatMap
import json

# Camera coordinates (your Saudi locations)
camera_positions = {
    "CAM_01": (24.687060, 46.708943),
    "CAM_02": (24.668566, 46.695156),
    "CAM_03": (24.665760, 46.682430),
}

# Load status.json from detection folder
with open("../detection/status.json", "r") as f:
    status = json.load(f)

cameras = []
for cam_id, info in status.items():
    lat, lon = camera_positions.get(cam_id, (None, None))
    if lat is None:
        continue
    cameras.append({
        "id": cam_id,
        "name": info["name"],
        "lat": lat,
        "lon": lon,
        "vehicles": info["vehicles"],
        "level": info["level"]
    })

# Center map
avg_lat = sum(c["lat"] for c in cameras) / len(cameras)
avg_lon = sum(c["lon"] for c in cameras) / len(cameras)

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)


def get_color(level):
    return {"Low": "green", "Medium": "orange", "High": "red"}[level]


# Camera markers
for cam in cameras:
    popup = f"{cam['name']}<br>Vehicles: {cam['vehicles']}<br>Level: {cam['level']}"
    folium.CircleMarker(
        location=[cam["lat"], cam["lon"]],
        radius=10,
        color=get_color(cam["level"]),
        fill=True,
        fill_opacity=0.9,
        popup=popup
    ).add_to(m)

# Heatmap
heat_data = [[c["lat"], c["lon"], c["vehicles"]] for c in cameras]
HeatMap(heat_data, radius=30, blur=20).add_to(m)

m.save("smartflow_map.html")
print("Map generated → smartflow_map.html")
