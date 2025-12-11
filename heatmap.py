import json
import folium
from folium.plugins import HeatMap

STATUS_FILE = "status.json"

with open(STATUS_FILE, "r", encoding="utf-8") as f:
    status = json.load(f)

cameras = []
for cam_id, info in status.items():
    cameras.append({
        "id": cam_id,
        "name": info["name"],
        "lat": info["lat"],
        "lon": info["lon"],
        "vehicles": info["vehicles"],
        "level": info["level"],
        "incidents": info.get("incidents", 0)
    })

# Center map
avg_lat = sum(c["lat"] for c in cameras) / len(cameras)
avg_lon = sum(c["lon"] for c in cameras) / len(cameras)

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="OpenStreetMap")

def color_for_level(level):
    return {"Low Traffic": "green",
            "Medium Traffic": "orange",
            "High Traffic": "red"}.get(level, "blue")

# Markers
for cam in cameras:
    popup = f"{cam['name']}<br>Vehicles: {cam['vehicles']}<br>Level: {cam['level']}<br>Incidents: {cam['incidents']}"
    folium.CircleMarker(
        location=[cam["lat"], cam["lon"]],
        radius=10,
        color=color_for_level(cam["level"]),
        fill=True,
        fill_opacity=0.9,
        popup=popup
    ).add_to(m)

# Heatmap weighted by vehicles
heat_data = [[cam["lat"], cam["lon"], cam["vehicles"]] for cam in cameras]
HeatMap(heat_data, radius=25, blur=15).add_to(m)

m.save("smartflow_heatmap.html")
print("✅ Heatmap generated: smartflow_heatmap.html")
