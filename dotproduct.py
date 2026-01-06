import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, TextBox, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ================= FIGURE & LAYOUT =================
fig = plt.figure(figsize=(16,11))
ax = fig.add_subplot(projection='3d')

title_daya = fig.text(
    0.50, 0.93,            
    "Daya: P = F · v",
    fontsize=16,
    weight="bold",
    ha="center"
)

plt.subplots_adjust(
    left=0.22,
    right=0.80,
    bottom=0.14,
    top=0.82
)

# ================= TEXTBOX INPUT =================
ax_tv_mag   = plt.axes([0.30, 0.84, 0.08, 0.04])
ax_tv_theta = plt.axes([0.40, 0.84, 0.08, 0.04])
ax_tF_mag   = plt.axes([0.50, 0.84, 0.08, 0.04])
ax_tF_theta = plt.axes([0.60, 0.84, 0.08, 0.04])

tv_mag   = TextBox(ax_tv_mag, "|v|")
tv_theta = TextBox(ax_tv_theta, "θ_v")
tF_mag   = TextBox(ax_tF_mag, "|F|")
tF_theta = TextBox(ax_tF_theta, "θ_F")

tv_mag.set_val("10")
tv_theta.set_val("0")
tF_mag.set_val("10")
tF_theta.set_val("90")

# ================= RADIO BUTTON =================
ax_v_dir = plt.axes([0.02, 0.50, 0.16, 0.25])
ax_F_dir = plt.axes([0.02, 0.18, 0.16, 0.25])

radio_v = RadioButtons(ax_v_dir, ('+X','-X','+Y','-Y','+Z','-Z'), active=0)
radio_F = RadioButtons(ax_F_dir, ('+X','-X','+Y','-Y','+Z','-Z'), active=2)

fig.text(0.04, 0.77, "Komponen V", fontsize=11, weight="bold")
fig.text(0.04, 0.45, "Komponen F", fontsize=11, weight="bold")

# ================= SLIDER CAMERA =================
ax_cam  = plt.axes([0.30, 0.08, 0.50, 0.03])
ax_zoom = plt.axes([0.30, 0.04, 0.50, 0.03])

s_cam  = Slider(ax_cam, "Rotasi kanan–kiri (deg)", -180, 180, valinit=45)
s_zoom = Slider(ax_zoom, "Zoom", 0.5, 3.0, valinit=1.0)

# ================= RESET =================
ax_reset = plt.axes([0.02, 0.08, 0.16, 0.05])
btn_reset = Button(ax_reset, "Reset View")

# ================= INFO BOX =================
fig.text(0.82, 0.77, "Informasi Vektor", fontsize=12, weight="bold")

bubble = fig.text(
    0.82, 0.60, "",
    fontsize=11,
    family="monospace",
    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85)
)

# ================= VEKTOR FUNCTIONS =================
def dir_to_vector(label):
    return {
        '+X':[1,0,0], '-X':[-1,0,0],
        '+Y':[0,1,0], '-Y':[0,-1,0],
        '+Z':[0,0,1], '-Z':[0,0,-1]
    }[label]

def rotate(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    return (v*np.cos(ang)
            + np.cross(axis,v)*np.sin(ang)
            + axis*np.dot(axis,v)*(1-np.cos(ang)))

def apply_rotation(base, theta):
    ref = np.array([0,0,1]) if abs(base[2]) < 0.9 else np.array([1,0,0])
    axis = np.cross(base, ref)
    axis = axis / np.linalg.norm(axis)
    return rotate(base, axis, theta)

# ================= UPDATE =================
def update(val=None):
    ax.cla()

    v_mag = float(tv_mag.text)
    F_mag = float(tF_mag.text)
    v_th  = np.deg2rad(float(tv_theta.text))
    F_th  = np.deg2rad(float(tF_theta.text))

    v_dir = apply_rotation(np.array(dir_to_vector(radio_v.value_selected)), v_th)
    F_dir = apply_rotation(np.array(dir_to_vector(radio_F.value_selected)), F_th)

    v = v_mag * v_dir
    F = F_mag * F_dir
    P = np.dot(F, v)

    ax.quiver(0,0,0,*v, color='b', linewidth=3, label='v')
    ax.quiver(0,0,0,*F, color='r', linewidth=3, label='F')

    verts = [[(0,0,0), tuple(v), tuple(v+F), tuple(F)]]
    ax.add_collection3d(
        Poly3DCollection(verts, facecolor='purple', alpha=0.35)
    )

    lim = max(np.linalg.norm(v), np.linalg.norm(F)) + 2
    lim /= s_zoom.val

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.view_init(elev=22, azim=s_cam.val)

    magF = np.linalg.norm(F)
    magv = np.linalg.norm(v)
    cosθ = P/(magF*magv) if magF and magv else 0
    θ = np.degrees(np.arccos(np.clip(cosθ,-1,1)))

    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    bubble.set_text(
        f"v = ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}) m/s\n"
        f"F = ({F[0]:.2f}, {F[1]:.2f}, {F[2]:.2f}) N\n"
        f"P = {P:.2f} W\n"
        f"θ(F,v) = {θ:.1f}°"
    )

    fig.canvas.draw_idle()

# ================= EVENTS =================
for tb in [tv_mag, tv_theta, tF_mag, tF_theta]:
    tb.on_submit(update)

radio_v.on_clicked(update)
radio_F.on_clicked(update)
s_cam.on_changed(update)
s_zoom.on_changed(update)

btn_reset.on_clicked(lambda e: update())

update()
plt.show()