import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# Color constants for clinical status
STATUS_COLORS = {
    "Stenotic": "#ef4444",      # Clinical red for stenosis alert
    "Normal": "#10b981",        # Emerald green for normal patency
    "Not assessed": "#64748b",  # Cool slate grey for unassessed
}

VESSEL_METADATA = {
    "LAD": {
        "full_name": "Left Anterior Descending Artery",
        "short_desc": "Anterior Interventricular Artery",
        "territory": "Anterior ventricular wall, anterior 2/3 of interventricular septum, and apex",
        "clinical_significance": "Supplies 45-55% of left ventricular myocardium; critical for systolic ejection fraction.",
    },
    "LCX": {
        "full_name": "Left Circumflex Artery",
        "short_desc": "Circumflex Branch of Left Coronary Artery",
        "territory": "Lateral wall and posterior-lateral aspects of the left ventricle",
        "clinical_significance": "Supplies lateral and posterior LV myocardium; stenosis impairs lateral wall contractility.",
    },
    "RCA": {
        "full_name": "Right Coronary Artery",
        "short_desc": "Main Right Coronary Trunk & Branches",
        "territory": "Right ventricle, inferior wall of left ventricle, SA & AV nodes (in 90% right-dominant hearts)",
        "clinical_significance": "Supplies cardiac conduction system and inferior wall; stenosis carries high AV block risk.",
    }
}


def generate_anatomical_heart_mesh(resolution: int = 50):
    """
    Generate an anatomical 3D ventricular mesh with physiological left/right ventricle asymmetry,
    anterior interventricular sulcus depression, and apical tapering.
    """
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(-np.pi / 2, np.pi / 2, resolution)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Ventricular geometry with physiological contours
    r_base = 1.0 - 0.22 * np.sin(phi_grid)
    # Left ventricle prominence and anterior groove indentation
    sulcus_indent = 0.08 * np.sin(2 * theta_grid) * np.cos(phi_grid)
    
    x = (1.20 * r_base + sulcus_indent) * np.cos(phi_grid) * np.sin(theta_grid)
    y = (0.95 * r_base) * np.cos(phi_grid) * np.cos(theta_grid)
    # Apex tapering toward negative z
    z = 1.35 * np.sin(phi_grid) - 0.25 * (1.0 - np.sin(phi_grid)) * np.sin(theta_grid)

    return x, y, z


def create_heart_figure(
    vessel_status: Optional[Dict[str, Any]] = None,
    show_orientation_labels: bool = True,
    theme: str = "dark"
) -> go.Figure:
    """
    Build an interactive, professional 3D coronary artery anatomical visualization.
    """
    vessel_status = vessel_status or {}

    def parse_vessel(name: str):
        val = vessel_status.get(name, "Not assessed")
        if isinstance(val, dict):
            status = val.get("status", "Not assessed")
            prob = val.get("probability", None)
        else:
            status = str(val)
            prob = None

        if status not in STATUS_COLORS:
            status = "Stenotic" if "sten" in status.lower() else ("Normal" if "norm" in status.lower() else "Not assessed")
        return status, prob

    lad_status, lad_prob = parse_vessel("LAD")
    lcx_status, lcx_prob = parse_vessel("LCX")
    rca_status, rca_prob = parse_vessel("RCA")

    # Generate 3D ventricular myocardium
    X, Y, Z = generate_anatomical_heart_mesh(resolution=45)

    fig = go.Figure()

    # 1. Ventricular Myocardium Surface
    # Sleek dark medical glassmorphic texture
    myo_colorscale = [
        [0.0, "#1e0b11"],
        [0.3, "#3b1219"],
        [0.7, "#581b24"],
        [1.0, "#732431"],
    ] if theme == "dark" else [
        [0.0, "#ffe4e6"],
        [0.5, "#fecdd3"],
        [1.0, "#fda4af"],
    ]

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=myo_colorscale,
            showscale=False,
            opacity=0.88,
            hoverinfo="skip",
            name="Ventricular Myocardium",
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                fresnel=0.4,
                specular=0.55,
                roughness=0.35
            ),
            lightposition=dict(x=150, y=200, z=180)
        )
    )

    # 2. Ascending Aorta & Aortic Root Base
    u_aorta = np.linspace(0, 2 * np.pi, 30)
    r_aorta = 0.26
    z_aorta_levels = [1.20, 1.35, 1.50]
    for z_lvl in z_aorta_levels:
        fig.add_trace(
            go.Scatter3d(
                x=r_aorta * np.cos(u_aorta) + 0.04,
                y=r_aorta * np.sin(u_aorta) + 0.16,
                z=np.full_like(u_aorta, z_lvl),
                mode="lines",
                line=dict(color="#64748b" if theme == "dark" else "#94a3b8", width=3),
                hoverinfo="name",
                name="Aortic Root / Sinus of Valsalva",
                showlegend=False
            )
        )

    # 3. Left Main (LM) Coronary Trunk
    fig.add_trace(
        go.Scatter3d(
            x=[0.04, -0.02, -0.08],
            y=[0.24, 0.32, 0.38],
            z=[1.26, 1.22, 1.18],
            mode="lines",
            line=dict(color="#cbd5e1" if theme == "dark" else "#475569", width=10),
            name="Left Main (LM) Trunk",
            hovertemplate="<b>Left Main Coronary Artery (LM)</b><br>Bifurcates into LAD and LCX<extra></extra>",
            showlegend=False
        )
    )

    # 4. Detailed Coronary Vascular Tree with Bifurcations
    vessel_data = {
        "LAD": {
            # Main LAD stem down the anterior interventricular groove
            "main": {
                "x": [-0.08, 0.05, 0.18, 0.28, 0.35, 0.38, 0.32, 0.18],
                "y": [0.38, 0.48, 0.56, 0.58, 0.52, 0.38, 0.15, -0.05],
                "z": [1.18, 1.05, 0.88, 0.65, 0.30, -0.10, -0.55, -0.95],
            },
            # Diagonal Branch D1
            "branch1": {
                "x": [0.18, 0.28, 0.36],
                "y": [0.56, 0.62, 0.58],
                "z": [0.88, 0.72, 0.50],
            },
            # Diagonal Branch D2
            "branch2": {
                "x": [0.35, 0.44, 0.48],
                "y": [0.52, 0.44, 0.28],
                "z": [0.30, 0.10, -0.15],
            },
            "status": lad_status,
            "prob": lad_prob,
            "color": STATUS_COLORS[lad_status],
        },
        "LCX": {
            # Main LCX traversing the left atrioventricular groove
            "main": {
                "x": [-0.08, -0.22, -0.48, -0.68, -0.74, -0.65],
                "y": [0.38, 0.44, 0.38, 0.18, -0.15, -0.45],
                "z": [1.18, 1.08, 0.90, 0.65, 0.35, 0.08],
            },
            # Obtuse Marginal Branch OM1
            "branch1": {
                "x": [-0.48, -0.58, -0.62],
                "y": [0.38, 0.28, 0.12],
                "z": [0.90, 0.65, 0.35],
            },
            # Obtuse Marginal Branch OM2
            "branch2": {
                "x": [-0.68, -0.75, -0.72],
                "y": [0.18, 0.02, -0.18],
                "z": [0.65, 0.38, 0.10],
            },
            "status": lcx_status,
            "prob": lcx_prob,
            "color": STATUS_COLORS[lcx_status],
        },
        "RCA": {
            # Main RCA trunk descending right AV sulcus
            "main": {
                "x": [0.08, 0.32, 0.58, 0.68, 0.60, 0.42, 0.15],
                "y": [0.22, 0.15, -0.10, -0.38, -0.62, -0.68, -0.55],
                "z": [1.24, 1.02, 0.82, 0.52, 0.15, -0.25, -0.65],
            },
            # Acute Marginal Branch
            "branch1": {
                "x": [0.58, 0.66, 0.62],
                "y": [-0.10, -0.20, -0.35],
                "z": [0.82, 0.55, 0.25],
            },
            # Posterior Descending Artery (PDA)
            "branch2": {
                "x": [0.42, 0.32, 0.20],
                "y": [-0.68, -0.55, -0.32],
                "z": [-0.25, -0.45, -0.68],
            },
            "status": rca_status,
            "prob": rca_prob,
            "color": STATUS_COLORS[rca_status],
        }
    }

    for v_name, data in vessel_data.items():
        meta = VESSEL_METADATA[v_name]
        status_label = data["status"]
        prob_str = f"{data['prob'] * 100:.1f}%" if data["prob"] is not None else "N/A"

        hover_html = (
            f"<b>{v_name} — {meta['full_name']}</b><br>"
            f"Predicted Status: <b>{status_label}</b><br>"
            f"Model Likelihood: <b>{prob_str}</b><br>"
            f"Supplied Territory: {meta['territory']}<br>"
            f"<i>{meta['clinical_significance']}</i>"
            "<extra></extra>"
        )

        # 1. Main Arterial Stem
        fig.add_trace(
            go.Scatter3d(
                x=data["main"]["x"],
                y=data["main"]["y"],
                z=data["main"]["z"],
                mode="lines",
                name=f"{v_name}: {status_label}",
                line=dict(color=data["color"], width=10),
                hovertemplate=hover_html,
                legendgroup=v_name,
            )
        )

        # 2. Side Branches (Diagonals / Marginals)
        for b_key in ["branch1", "branch2"]:
            fig.add_trace(
                go.Scatter3d(
                    x=data[b_key]["x"],
                    y=data[b_key]["y"],
                    z=data[b_key]["z"],
                    mode="lines",
                    line=dict(color=data["color"], width=5),
                    hovertemplate=hover_html,
                    legendgroup=v_name,
                    showlegend=False
                )
            )

        # 3. Clinical Inspection Node Markers
        fig.add_trace(
            go.Scatter3d(
                x=data["main"]["x"],
                y=data["main"]["y"],
                z=data["main"]["z"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=data["color"],
                    symbol="circle",
                    line=dict(color="#ffffff", width=1.2),
                ),
                hovertemplate=hover_html,
                legendgroup=v_name,
                showlegend=False,
            )
        )

    # 5. Anatomical Compass Orientation Labels
    if show_orientation_labels:
        orientations = [
            {"text": "Superior (Aorta Base)", "x": 0.04, "y": 0.16, "z": 1.58, "color": "#38bdf8"},
            {"text": "Inferior (Cardiac Apex)", "x": 0.18, "y": -0.05, "z": -1.15, "color": "#38bdf8"},
            {"text": "Anterior Wall", "x": 0.20, "y": 0.85, "z": 0.35, "color": "#94a3b8"},
            {"text": "Lateral / Posterior", "x": -0.75, "y": -0.30, "z": 0.25, "color": "#94a3b8"},
        ]
        fig.add_trace(
            go.Scatter3d(
                x=[o["x"] for o in orientations],
                y=[o["y"] for o in orientations],
                z=[o["z"] for o in orientations],
                mode="text",
                text=[o["text"] for o in orientations],
                textposition="top center",
                textfont=dict(size=11, color="#cbd5e1" if theme == "dark" else "#334155"),
                hoverinfo="text",
                showlegend=False,
            )
        )

    bg_color = "#090d16" if theme == "dark" else "#f8fafc"
    font_color = "#f8fafc" if theme == "dark" else "#0f172a"

    fig.update_layout(
        title=dict(
            text="<b>3D Epicardial Coronary Tree & Ventricular Perfusion</b>",
            font=dict(size=14, color=font_color),
            x=0.03,
            y=0.96,
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=1.1, y=1.0, z=0.9),
            bgcolor=bg_color,
            camera=dict(
                eye=dict(x=1.40, y=1.40, z=0.85),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.05),
            ),
        ),
        paper_bgcolor=bg_color,
        font=dict(family="Inter, system-ui, sans-serif", color=font_color),
        height=540,
        margin=dict(l=10, r=10, t=50, b=20),
        hoverlabel=dict(
            bgcolor="#0b1329",
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#ffffff"),
            bordercolor="#38bdf8",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.03,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15, 23, 42, 0.8)" if theme == "dark" else "rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(148, 163, 184, 0.3)",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    return fig


def get_vessel_status_dict(predictions: dict) -> Dict[str, Any]:
    """Helper to convert predict_patient output dictionary into vessel_status format for 3D view."""
    vessel_status = {}
    for vessel in ["lad", "lcx", "rca"]:
        if vessel in predictions:
            res = predictions[vessel]
            status_label = "Stenotic" if res.get("predicted_class") == 1 else "Normal"
            vessel_status[vessel.upper()] = {
                "status": status_label,
                "probability": res.get("positive_probability"),
            }
        else:
            vessel_status[vessel.upper()] = {
                "status": "Not assessed",
                "probability": None,
            }
    return vessel_status