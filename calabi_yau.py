import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def plot_calabi_yau(elev, azim, alpha_deg):
    # 1. Set up the plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    equation_text = (
        r"$\bf{Calabi-Yau\ Cross\ Section}$"
        + "\n"
        + r"$z_1^5 + z_2^5 = 1$"
        + "\n"
        + r"Proj: $Z = \cos(\alpha)\mathrm{Im}(z_1) + \sin(\alpha)\mathrm{Im}(z_2)$"
    )
    fig.text(
        0.02,
        0.92,
        equation_text,
        color="white",
        fontsize=14,
        family="monospace",
        verticalalignment="top",
    )

    param_text = (
        f"Projection Alpha: {alpha_deg:>6.1f}°\n"
        f"Camera Azimuth:   {azim:>6.1f}°\n"
        f"Camera Elevation: {elev:>6.1f}°"
    )
    fig.text(
        0.02,
        0.82,
        param_text,
        color="cyan",
        fontsize=12,
        family="monospace",
        verticalalignment="top",
    )

    n = 5
    alpha = np.radians(alpha_deg)
    xi_steps = 25
    theta_steps = 25

    xi = np.linspace(-1.0, 1.0, xi_steps)
    theta = np.linspace(0, np.pi / 2, theta_steps)
    Xi, Theta = np.meshgrid(xi, theta)

    all_x, all_y, all_z = [], [], []

    print(f"Generating: Elev={elev}, Azim={azim}, Alpha={alpha_deg}")

    for k1 in range(n):
        for k2 in range(n):
            phase1 = np.exp(2j * np.pi * k1 / n)
            phase2 = np.exp(2j * np.pi * k2 / n)

            w = Xi + 1j * Theta
            z1 = phase1 * (np.cosh(w)) ** (2 / n)
            z2 = phase2 * (np.sinh(w)) ** (2 / n)

            X = z1.real
            Y = z2.real
            Z = np.cos(alpha) * z1.imag + np.sin(alpha) * z2.imag

            all_x.append(X)
            all_y.append(Y)
            all_z.append(Z)

            for i in range(len(xi)):
                ax.plot(
                    X[:, i], Y[:, i], Z[:, i], color="white", alpha=0.15, linewidth=0.6
                )
            for i in range(len(theta)):
                ax.plot(
                    X[i, :], Y[i, :], Z[i, :], color="white", alpha=0.15, linewidth=0.6
                )

    full_x = np.concatenate([x.flatten() for x in all_x])
    full_y = np.concatenate([y.flatten() for y in all_y])
    full_z = np.concatenate([z.flatten() for z in all_z])

    max_range = (
        np.array(
            [
                full_x.max() - full_x.min(),
                full_y.max() - full_y.min(),
                full_z.max() - full_z.min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (full_x.max() + full_x.min()) * 0.5
    mid_y = (full_y.max() + full_y.min()) * 0.5
    mid_z = (full_z.max() + full_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect((1, 1, 1))

    ax.view_init(elev=elev, azim=azim)

    filename = (
        f"images/calabi_yau_elev{int(elev)}_azim{int(azim)}_alpha{int(alpha_deg)}.png"
    )
    plt.savefig(filename, dpi=100)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--elev", type=float, default=32.0)
    parser.add_argument("--azim", type=float, default=45.0)
    parser.add_argument("--alpha", type=float, default=45.0)
    args = parser.parse_args()

    if args.all:
        for azim in range(0, 360, 15):
            for alpha in range(0, 120, 30):
                plot_calabi_yau(32, azim, alpha)
    else:
        plot_calabi_yau(args.elev, args.azim, args.alpha)
