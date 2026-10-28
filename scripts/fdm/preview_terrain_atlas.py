#!/usr/bin/env python3
"""Build the FDM mesh atlas and export offline previews without Isaac Sim.

python scripts/fdm/preview_terrain_atlas.py --seed 0
Requires numpy, trimesh, scipy, matplotlib and plotly in the active environment.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "unitree_rl_lab"))

import numpy as np

from unitree_rl_lab.fdm.terrain_atlas import (
    Atlas, COLORS, SIZE, build_atlas, check_geometry,
)


def _polygons(mesh):
    """Merge coplanar triangles for clean Matplotlib rendering of stair sides."""
    used = set()
    result = []
    for facet, boundary in zip(mesh.facets, mesh.facets_boundary):
        neighbors = {}
        for a, b in boundary:
            neighbors.setdefault(int(a), []).append(int(b))
            neighbors.setdefault(int(b), []).append(int(a))
        if not neighbors or any(len(adj) != 2 for adj in neighbors.values()):
            continue
        start = min(neighbors)
        order, previous, current = [start], None, start
        while True:
            nxt = next(v for v in neighbors[current] if v != previous)
            if nxt == start:
                break
            if nxt in order:
                break
            order.append(nxt)
            previous, current = current, nxt
        if len(order) != len(neighbors):
            continue  # Facet with holes: keep its original triangles.
        result.append((mesh.vertices[order], mesh.face_normals[facet[0]]))
        used.update(int(i) for i in facet)
    result.extend((mesh.triangles[i], mesh.face_normals[i])
                  for i in range(len(mesh.faces)) if i not in used)
    return result


def _configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["savefig.facecolor"] = "#f7f8fa"


def _draw_3d(ax, atlas, zone=None):
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if zone:
        cx, cy = zone["center"]
        bounds = (cx - 6, cx + 6, cy - 6, cy + 6)
    else:
        bounds = (-24, 24, -18, 18)
    xmin, xmax, ymin, ymax = bounds
    floor = [[(xmin, ymin, 0), (xmax, ymin, 0), (xmax, ymax, 0), (xmin, ymax, 0)]]
    ax.add_collection3d(Poly3DCollection(floor, facecolors=COLORS["floor"],
                                       edgecolors="#b9c2c8", linewidths=0.4, zorder=1))
    polygons, colors = [], []
    light = np.array([-0.4, -0.5, 0.85])
    light /= np.linalg.norm(light)
    for part in atlas.parts:
        if part.kind == "floor" or (zone and part.zone != zone["id"]):
            continue
        base = np.array(to_rgb(COLORS[part.kind]))
        for polygon, normal in _polygons(part.mesh):
            polygons.append(polygon)
            colors.append(np.clip(base * (0.70 + 0.30 * max(0, normal @ light)), 0, 1))
    ax.add_collection3d(Poly3DCollection(
        polygons, facecolors=np.asarray(colors), edgecolors=(0.13, 0.20, 0.27, 0.24),
        linewidths=0.22 if zone else 0.12, zsort="average", zorder=2,
    ))
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(-0.1, 1.8))
    # Equal metric scale. Do not exaggerate stair heights in exported figures.
    ax.set_box_aspect((xmax - xmin, ymax - ymin, 1.9))
    ax.view_init(elev=40 if zone else 53, azim=-65)
    ax.set_axis_off()
    if not zone:
        for z in atlas.zones:
            cx, cy = z["center"]
            ax.text(cx, cy + 4.8, 0.15, z["id"], fontsize=9, color="#263c50",
                    ha="center", zorder=4)
    else:
        for key, color, label in (("start_ground_xyz", "#16866c", "S"),
                                  ("goal_ground_xyz", "#c95b53", "G")):
            x, y, _ = zone["test"][key]
            ax.scatter(x, y, 0.08, c=color, s=24, depthshade=False, zorder=4)
            ax.text(x, y, 0.25, label, color=color, fontsize=8, zorder=4)


def export_pngs(atlas, output):
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.colors import to_rgb

    fig, ax = plt.subplots(figsize=(16, 12.5), facecolor="#f7f8fa")
    ax.set_facecolor("#e9edf0")
    for i, zone in enumerate(atlas.zones):
        (x0, y0), (x1, y1) = zone["bounds_xy"]
        ax.add_patch(Rectangle((x0, y0), 12, 12, facecolor=("#e7ecee", "#edf0f2")[i % 2],
                               edgecolor="#a5b2bb", linewidth=0.7, linestyle="--"))
    for part in atlas.parts:
        if part.kind == "floor":
            continue
        faces = part.mesh.face_normals[:, 2] > 0.01
        triangles = part.mesh.triangles[faces]
        normals = part.mesh.face_normals[faces]
        base = np.array(to_rgb(COLORS[part.kind]))
        shades = (0.82 + 0.18 * normals[:, 2])[:, None]
        ax.add_collection(PolyCollection(
            triangles[:, :, :2], facecolors=shades * base, edgecolors="none", antialiased=False))
    for zone in atlas.zones:
        cx, cy = zone["center"]
        ax.text(cx - 5.3, cy + 5.1, f"{zone['id']}  {zone['title']}", fontsize=10,
                color="#23384a", weight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f7f8fa", ec="none", alpha=0.94))
    ax.scatter(*atlas.spawn[:2], s=90, marker="*", color="#178567", edgecolor="white", zorder=8)
    ax.annotate("起点", atlas.spawn[:2], xytext=(-18, -18), textcoords="offset points",
                fontsize=9, color="#176c58")
    for port in atlas.ports:
        ax.plot(*port["ground_xyz"][:2], "o", color="#f7f8fa", markeredgecolor="#5d8c82",
                markersize=4, markeredgewidth=0.8)
    ax.set(xlim=(-24.4, 24.4), ylim=(-18.4, 18.4), xlabel="X / m", ylabel="Y / m")
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-24, 25, 6))
    ax.set_yticks(np.arange(-18, 19, 6))
    ax.tick_params(colors="#657985")
    for spine in ax.spines.values():
        spine.set_visible(False)
    labels = {"stairs": "楼梯", "ramp": "坡道", "platform": "平台",
              "wall": "墙体", "pillar": "柱体", "rough": "低矮起伏"}
    ax.legend(handles=[Patch(color=COLORS[k], label=v) for k, v in labels.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=6, frameon=False)
    fig.suptitle("FDM 复杂路面拼接地图", fontsize=24, x=0.12, ha="left", y=0.97, color="#20374b")
    fig.text(0.12, 0.935, f"48 × 36 m  ·  12 个相连分区  ·  seed {atlas.seed}  ·  圆点为检查过的分区接口",
             fontsize=12, color="#657985")
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.09, right=0.96)
    fig.savefig(output / "overview.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 11), facecolor="#f7f8fa")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    _draw_3d(ax, atlas)
    fig.suptitle("整图三维预览 · 实际几何比例", fontsize=21, y=0.96, color="#20374b")
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    fig.savefig(output / "perspective.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(20, 13), facecolor="#f7f8fa")
    for i, zone in enumerate(atlas.zones):
        ax = fig.add_subplot(3, 4, i + 1, projection="3d", computed_zorder=False)
        _draw_3d(ax, atlas, zone)
        ax.set_title(f"{zone['id']}  {zone['title']}", fontsize=12, color="#20374b", pad=-10)
    fig.suptitle("分区近景 · S 起点 / G 目标点", fontsize=22, y=0.99, color="#20374b")
    fig.text(0.5, 0.955, "显示各分区内部障碍；接缝挡墙见整图。点位用于后续实验，不代表已经验证机器人通行。",
             ha="center", fontsize=11, color="#657985")
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0.01, wspace=0, hspace=0.02)
    fig.savefig(output / "details.png", dpi=150)
    plt.close(fig)


def _plotly_figure(atlas: Atlas, zone=None):
    import plotly.graph_objects as go

    traces = []
    if zone:
        cx, cy = zone["center"]
        xmin, xmax, ymin, ymax = cx - 6, cx + 6, cy - 6, cy + 6
        traces.append(go.Mesh3d(x=[xmin, xmax, xmax, xmin], y=[ymin, ymin, ymax, ymax],
                               z=[0] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
                               color=COLORS["floor"], hoverinfo="skip", name="Ground"))
    else:
        xmin, xmax, ymin, ymax = -24, 24, -18, 18
    for part in atlas.parts:
        if zone and part.zone != zone["id"]:
            continue
        vertices, faces = part.mesh.vertices, part.mesh.faces
        traces.append(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=COLORS[part.kind], flatshading=True, name=part.name,
            hovertemplate=f"{part.name}<br>X %{{x:.2f}} m · Y %{{y:.2f}} m<br>Z %{{z:.2f}} m<extra></extra>",
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.12, roughness=0.9),
            lightposition=dict(x=-100, y=-150, z=250),
        ))
    if zone:
        for key, color, symbol, label in (
            ("start_ground_xyz", "#16866c", "circle", "S · 起点"),
            ("goal_ground_xyz", "#c95b53", "diamond", "G · 目标"),
        ):
            x, y, _ = zone["test"][key]
            traces.append(go.Scatter3d(x=[x], y=[y], z=[0.12], text=[label],
                                      mode="markers+text", textposition="top center",
                                      marker=dict(size=5, color=color, symbol=symbol),
                                      name=label, hoverinfo="text"))
    else:
        for zone_info in atlas.zones:
            cx, cy = zone_info["center"]
            traces.append(go.Scatter3d(
                x=[cx], y=[cy + 5.1], z=[0.2], text=[f"{zone_info['id']} {zone_info['title']}"],
                mode="text", textfont=dict(size=11, color="#23384a"), hoverinfo="skip",
                name="分区标注",
            ))
        # Dashed zone boundaries are preview annotations, not mesh geometry.
        for x in (-12, 0, 12):
            traces.append(go.Scatter3d(x=[x, x], y=[-18, 18], z=[0.015, 0.015],
                                      mode="lines", line=dict(color="#b2bdc5", width=2, dash="dash"),
                                      hoverinfo="skip", name="分区线"))
        for y in (-6, 6):
            traces.append(go.Scatter3d(x=[-24, 24], y=[y, y], z=[0.015, 0.015],
                                      mode="lines", line=dict(color="#b2bdc5", width=2, dash="dash"),
                                      hoverinfo="skip", name="分区线"))
    scene = dict(
        xaxis=dict(title="X / m", range=[xmin, xmax], showbackground=False, gridcolor="#dce2e7"),
        yaxis=dict(title="Y / m", range=[ymin, ymax], showbackground=False, gridcolor="#dce2e7"),
        zaxis=dict(title="Z / m", range=[-0.15, 1.8], showbackground=False, gridcolor="#dce2e7"),
        aspectmode="data",
        camera=dict(eye=dict(x=0.9, y=-1.15, z=1.1), projection=dict(type="orthographic")),
        dragmode="orbit",
    )
    return go.Figure(traces, layout=dict(
        scene=scene, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        paper_bgcolor="#f2f5f7", font=dict(family="sans-serif", color="#23384a"),
    ))


def export_html(atlas, output, report):
    from plotly.utils import PlotlyJSONEncoder

    figure = _plotly_figure(atlas)
    plot = figure.to_html(full_html=False, include_plotlyjs=True, div_id="terrain-view",
                          config={"responsive": True, "displaylogo": False, "scrollZoom": True,
                                  "toImageButtonOptions": {"filename": "fdm_terrain_atlas", "scale": 2}})
    views = {"all": figure.to_plotly_json()}
    views.update({z["id"]: _plotly_figure(atlas, z).to_plotly_json() for z in atlas.zones})
    buttons = '<button class="zone active" data-zone="all">整张地图</button>' + "".join(
        f'<button class="zone" data-zone="{z["id"]}">{z["id"]} {html.escape(z["title"])}</button>'
        for z in atlas.zones)
    payload = json.dumps(views, cls=PlotlyJSONEncoder, ensure_ascii=False).replace("</", "<\\/")
    zones = json.dumps({z["id"]: z for z in atlas.zones}, ensure_ascii=False)
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FDM 复杂路面地图预览</title>
<style>
*{box-sizing:border-box}body{margin:0;color:#23384a;background:#edf1f4;font:15px system-ui,sans-serif}
header{background:#20374b;color:#fff;padding:22px 28px}h1{font-size:25px;margin:0 0 7px}
header p{margin:0;color:#bacbd8}main{padding:20px;display:grid;grid-template-columns:275px minmax(0,1fr);gap:18px}
aside,.panel{background:#fff;border:1px solid #dbe3e9;border-radius:12px;padding:16px}
.zones{display:grid;gap:6px;margin:14px 0}.zone,button{border:1px solid #d4dee5;background:#f7f9fa;
border-radius:7px;padding:9px 12px;color:#365063;cursor:pointer;text-align:left;font:inherit}
button:hover{background:#e5eef4}.active{background:#dcecf3;border-color:#78a8bc;color:#1c5269}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:9px}.panel{min-width:0}
#terrain-view{height:70vh;min-height:430px}.hint{font-size:13px;line-height:1.7;color:#6a7f8c}
#description{line-height:1.8;margin:8px 0}a{color:#247b96}h2{font-size:18px;margin:4px 0 10px}
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px}
.legend{display:flex;flex-wrap:wrap;gap:13px;margin:12px 0}.stats{font-size:13px;line-height:1.9}
@media(max-width:850px){main{grid-template-columns:1fr}.zones{grid-template-columns:repeat(2,minmax(0,1fr))}
#terrain-view{height:60vh}header{padding:18px}}
</style></head><body><header><h1>FDM 复杂路面拼接地图</h1>
<p>48 × 36 m · 4 × 3 分区 · 实际 mesh · 离线交互预览</p></header><main><aside>
<h2>选择区域</h2><div class="zones">__BUTTONS__</div>
<div class="stats">seed __SEED__ · __TRIANGLES__ 个三角面<br>17 个共享接口 · 42 个连通检查点<br>
几何检查：通过<br><a href="overview.png">俯视整图</a> · <a href="perspective.png">三维整图</a> ·
<a href="details.png">分区近景</a><br><a href="terrain.glb">GLB 模型</a> ·
<a href="terrain.ply">PLY 网格</a> · <a href="metadata.json">尺寸与点位</a></div>
<p class="hint">参考 FDM / nav-suite 的障碍设计重新组合。局部视图展示分区内部；接缝挡墙见整图。
本预览只验证几何，机器人通过能力需后续联合推理验证。</p>
</aside><section class="panel"><h2 id="view-title">整张地图</h2>
<div class="toolbar"><button id="oblique">三维视角</button><button id="top">俯视</button>
<button id="height">高度：真实比例</button><button id="reset">重置视角</button></div>
<p id="description">上排：起伏、柱阵与窄通道；中排：楼梯 / 坡道路线选择；下排：坡墙、多层平台、四向楼梯与迷宫。</p>
__PLOT__
<div class="legend"><span><i class="swatch" style="background:#558dbd"></i>楼梯</span>
<span><i class="swatch" style="background:#e4ad52"></i>坡道</span>
<span><i class="swatch" style="background:#54a79a"></i>平台</span>
<span><i class="swatch" style="background:#bb7767"></i>墙体</span>
<span><i class="swatch" style="background:#7c8b9c"></i>柱体</span>
<span><i class="swatch" style="background:#a293b8"></i>低矮起伏</span></div>
<div class="hint">拖动旋转 · 滚轮缩放 · 右键拖动平移 · 双击恢复。默认按真实比例显示；高度放大仅影响显示。
网格虚线、文字和 S / G 标记不属于碰撞地图。</div></section></main>
<script>
const views=__VIEWS__, zones=__ZONES__;
let selected='all', exaggerate=false;
const graph=document.getElementById('terrain-view');
function render(){
 const v=JSON.parse(JSON.stringify(views[selected]));
 if(exaggerate){v.layout.scene.aspectmode='manual';
   const s=selected==='all'?[48,36]:[12,12];
   v.layout.scene.aspectratio={x:s[0]/s[1],y:1,z:3*1.95/s[1]};}
 Plotly.react(graph,v.data,v.layout,{responsive:true,displaylogo:false,scrollZoom:true});
 document.querySelectorAll('[data-zone]').forEach(b=>b.classList.toggle('active',b.dataset.zone===selected));
 document.getElementById('view-title').textContent=selected==='all'?'整张地图':selected+' '+zones[selected].title;
 document.getElementById('description').textContent=selected==='all'?
 '上排：起伏、柱阵与窄通道；中排：楼梯 / 坡道路线选择；下排：坡墙、多层平台、四向楼梯与迷宫。':
 zones[selected].test.intent+'。S / G 是预设实验点位。';
 document.getElementById('height').textContent=exaggerate?'高度：放大 3 倍（仅显示）':'高度：真实比例';
}
document.querySelectorAll('[data-zone]').forEach(b=>b.onclick=()=>{selected=b.dataset.zone;render();});
document.getElementById('height').onclick=()=>{exaggerate=!exaggerate;render();};
document.getElementById('reset').onclick=render;
document.getElementById('oblique').onclick=()=>Plotly.relayout(graph,{'scene.camera':{
 eye:{x:.9,y:-1.15,z:1.1},up:{x:0,y:0,z:1},projection:{type:'orthographic'}}});
document.getElementById('top').onclick=()=>Plotly.relayout(graph,{'scene.camera':{
 eye:{x:0,y:0,z:2.2},up:{x:0,y:1,z:0},projection:{type:'orthographic'}}});
</script></body></html>"""
    replacements = {
        "__BUTTONS__": buttons, "__SEED__": str(atlas.seed),
        "__TRIANGLES__": str(report["triangles"]), "__PLOT__": plot,
        "__VIEWS__": payload, "__ZONES__": zones,
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    (output / "preview.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs" / "terrain_previews" / f"fdm_atlas_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    atlas = build_atlas(args.seed)
    report = check_geometry(atlas)
    (output / "geometry_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"Geometry validation failed: {report['failures']}")
    mesh = atlas.combined_mesh()
    mesh.export(output / "terrain.ply")
    mesh.export(output / "terrain.glb")
    metadata = atlas.metadata()
    metadata["checks"] = report
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    export_pngs(atlas, output)
    export_html(atlas, output, report)
    print(json.dumps({"output": str(output.resolve()), **report}, indent=2))


if __name__ == "__main__":
    main()
