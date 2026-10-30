import cv2, numpy as np, math, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import make_figures as mf
mf.style()
D = sys.argv[1] if len(sys.argv) > 1 else "."   # folder with frames/, depth_da2s/, depth_da2l/
W = 640; f = (0.5*W)/math.tan(math.radians(45))
u, v = np.meshgrid(np.arange(W)+0.5-W/2, np.arange(W)+0.5-W/2)
cosang = f/np.sqrt(u**2+v**2+f**2)
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(float)/1000.0
rows = [(98, "Feet visible (frame 98)"), (103, "Feet cut off (frame 103)")]
fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.8))
for r, (fi, lab) in enumerate(rows):
    cam = cv2.cvtColor(cv2.imread(f"{D}/frames/frame_{fi:05d}.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB)
    true = rd(f"{D}/frames/depth_{fi:05d}.png"); true[true <= 0] = np.nan
    s = rd(f"{D}/depth_da2s/depth_{fi:05d}.png")*cosang
    l = rd(f"{D}/depth_da2l/depth_{fi:05d}.png")*cosang
    rowmed = np.nanmedian(true, axis=1, keepdims=True)
    mask = (true < rowmed - 0.8) & (true < 10) & np.isfinite(true); mask[:250] = False
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    n, lab_im, stats, _ = cv2.connectedComponentsWithStats(mask)
    k = 1+np.argmax(stats[1:, cv2.CC_STAT_AREA]); mask = (lab_im == k).astype(np.uint8)
    m = mask.astype(bool)
    zt, zs, zl = np.nanmedian(true[m]), np.median(s[m]), np.median(l[m])
    print(fi, m.sum(), round(zt,2), round(zs,2), round(zl,2))
    ax = axes[r, 0]; ax.imshow(cam); ax.set_title(lab, loc="left", fontsize=10)
    ax = axes[r, 1]; im1 = ax.imshow(true, cmap="viridis", vmin=0, vmax=20)
    ax.set_title(f"Rendered: {zt:.2f} m", loc="left", fontsize=10)
    for c, (est, name, z) in enumerate([(s, "Small", zs), (l, "Large", zl)]):
        ax = axes[r, 2+c]
        im2 = ax.imshow(est-true, cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title(f"{name}: {z:.2f} m ({z-zt:+.2f})", loc="left", fontsize=10)
    for ax in axes[r, 1:]:
        ax.contour(mask, levels=[0.5], colors="black", linewidths=0.8)
    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(rect=(0, 0.1, 1, 0.92), h_pad=4.0, w_pad=0.8)
c1 = fig.add_axes([0.27, 0.07, 0.2, 0.02]); fig.colorbar(im1, cax=c1, orientation="horizontal").set_label("Rendered depth (m)", fontsize=10.5)
c2 = fig.add_axes([0.56, 0.07, 0.3, 0.02]); fig.colorbar(im2, cax=c2, orientation="horizontal").set_label("Estimate − rendered (m); red = too far, blue = too close", fontsize=10.5)
fig.suptitle("Depth error of Depth Anything V2 Small and Large (Baseline, 10 km/h; outline = person; values = person median)", x=0.01, y=0.99, ha="left", fontsize=10.5)
mf.save(fig, __import__("pathlib").Path(D), "F14_depth_failure_modes")
# numeric check at person
