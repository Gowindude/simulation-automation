"""Embed dashboard_data.json into dashboard.html as an inline script tag,
producing dashboard_publish.html ready to hand to the Artifact tool."""
import datetime
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "dashboard_data.json")) as f:
    data = json.load(f)
data["generated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

with open(os.path.join(BASE, "dashboard.html"), encoding="utf-8") as f:
    html = f.read()

script_tag = f'<script>window.__DASHBOARD_DATA__ = {json.dumps(data, separators=(",", ":"))};</script>\n'
html = html.replace("<script>\n(function", script_tag + "<script>\n(function")

out_path = os.path.join(BASE, "dashboard_publish.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
