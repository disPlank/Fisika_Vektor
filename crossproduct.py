# ===================== IMPORT LIBRARY =====================
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ===================== FIGURE & LAYOUT =====================
fig = plt.figure(figsize=(16,11))
ax = fig.add_subplot(projection='3d')

title_torsi = fig.text(
    0.50, 0.93,            
    "Torsi: τ = v × f",
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


# ===================== TEXTBOX INPUT =====================
ax_tr_mag   = plt.axes([0.30, 0.84, 0.08, 0.04])
ax_tr_theta = plt.axes([0.40, 0.84, 0.08, 0.04])
ax_tF_mag   = plt.axes([0.50, 0.84, 0.08, 0.04])
ax_tF_theta = plt.axes([0.60, 0.84, 0.08, 0.04])

tr_mag   = TextBox(ax_tr_mag, "|v|")
tr_theta = TextBox(ax_tr_theta, "θ_v")
tF_mag   = TextBox(ax_tF_mag, "|f|")
tF_theta = TextBox(ax_tF_theta, "θ_f")

tr_mag.set_val("10")
tr_theta.set_val("0")
tF_mag.set_val("10")
tF_theta.set_val("90")


# ===================== RADIO BUTTON =====================
ax_r_dir = plt.axes([0.02, 0.50, 0.16, 0.25])
ax_F_dir = plt.axes([0.02, 0.18, 0.16, 0.25])

radio_r = RadioButtons(ax_r_dir, ('+X','-X','+Y','-Y','+Z','-Z'), active=0)
radio_F = RadioButtons(ax_F_dir, ('+X','-X','+Y','-Y','+Z','-Z'), active=2)

fig.text(0.04, 0.77, "Komponen v", fontsize=11, weight="bold")
fig.text(0.04, 0.45, "Komponen f", fontsize=11, weight="bold")


# ===================== SLIDER CAMERA =====================
ax_cam  = plt.axes([0.30, 0.08, 0.50, 0.03])
ax_zoom = plt.axes([0.30, 0.04, 0.50, 0.03])

s_cam  = Slider(ax_cam, "Rotasi kanan–kiri (deg)", -180, 180, valinit=45)
s_zoom = Slider(ax_zoom, "Zoom", 0.5, 3.0, valinit=1.0)


# ===================== RESET BUTTON =====================
ax_reset = plt.axes([0.02, 0.08, 0.16, 0.05])
btn_reset = Button(ax_reset, "Reset View")

def reset_view(event):
    s_cam.set_val(45)
    s_zoom.set_val(1.0)

btn_reset.on_clicked(reset_view)


# ===================== INFO BOX =====================
fig.text(0.82, 0.77, "Informasi Vektor", fontsize=12, weight="bold")

bubble = fig.text(
    0.82, 0.60, "",
    fontsize=11,
    family="monospace",
    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85)
)


# ===================== FUNGSI BANTU =====================
def dir_to_vector(label):
    return {
        '+X': np.array([1,0,0]), '-X': np.array([-1,0,0]),
        '+Y': np.array([0,1,0]), '-Y': np.array([0,-1,0]),
        '+Z': np.array([0,0,1]), '-Z': np.array([0,0,-1])
    }[label]

def rotate(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    return (v*np.cos(ang)
            + np.cross(axis, v)*np.sin(ang)
            + axis*np.dot(axis, v)*(1-np.cos(ang)))

def apply_rotation(base_dir, theta):
    ref = np.array([0,0,1]) if abs(base_dir[2]) < 0.9 else np.array([1,0,0])
    axis = np.cross(base_dir, ref)
    axis = axis / np.linalg.norm(axis)
    return rotate(base_dir, axis, theta)


# ===================== UPDATE =====================
def update(val=None):
    ax.cla()

    r_mag = float(tr_mag.text)
    r_theta = np.deg2rad(float(tr_theta.text))
    F_mag = float(tF_mag.text)
    F_theta = np.deg2rad(float(tF_theta.text))

    r_dir = apply_rotation(dir_to_vector(radio_r.value_selected), r_theta)
    F_dir = apply_rotation(dir_to_vector(radio_F.value_selected), F_theta)

    r = r_mag * r_dir
    F = F_mag * F_dir
    tau = np.cross(r, F)

    ax.quiver(0,0,0,*r,color='g',linewidth=3,label='v')
    ax.quiver(0,0,0,*F,color='r',linewidth=3,label='f')
    ax.quiver(0,0,0,*tau,color='b',linewidth=3,label='τ')

    verts = [[(0,0,0), tuple(r), tuple(r+F), tuple(F)]]
    ax.add_collection3d(
        Poly3DCollection(verts, alpha=0.35, facecolor='purple'))

    base_lim = max(np.linalg.norm(r), np.linalg.norm(F), np.linalg.norm(tau)) + 1
    lim = base_lim / s_zoom.val
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.view_init(elev=22, azim=s_cam.val)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    bubble.set_text(
        f"v = ({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f})\n"
        f"f = ({F[0]:.2f}, {F[1]:.2f}, {F[2]:.2f})\n"
        f"τ = ({tau[0]:.2f}, {tau[1]:.2f}, {tau[2]:.2f})"
    )

    fig.canvas.draw_idle()


# ===================== CONNECT =====================
for w in [s_cam, s_zoom]:
    w.on_changed(update)

radio_r.on_clicked(update)
radio_F.on_clicked(update)

tr_mag.on_submit(lambda t: update())
tr_theta.on_submit(lambda t: update())
tF_mag.on_submit(lambda t: update())
tF_theta.on_submit(lambda t: update())

update()
plt.show()