import os
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 11


def plot_distance_fixed_angle(data_rows, fixed_angle=0, save_prefix="results/plots"):
    """
    Fixed angle, varying distance
    Left y-axis: R^r, P^r, F1^r as lines (0~1)
    Right y-axis: Acc (%) as bars (0~100)
    """
    os.makedirs(save_prefix, exist_ok=True)

    rows = [r for r in data_rows if r["angle"] == fixed_angle]
    rows.sort(key=lambda x: x["distance"])

    x = [r["distance"] for r in rows]
    R = [r["R"] for r in rows]
    P = [r["P"] for r in rows]
    F1 = [r["F1"] for r in rows]
    Acc = [r["Acc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Lines (left axis)
    ax1.plot(x, R, marker="o", linewidth=2, label=r"$R^{r}$")
    ax1.plot(x, P, marker="o", linewidth=2, label=r"$P^{r}$")
    ax1.plot(x, F1, marker="o", linewidth=2, label=r"$F_{1}^{r}$")
    ax1.set_xlabel("Distance (cm)")
    ax1.set_ylabel(r"$R^{r},\ P^{r},\ F_{1}^{r}$")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    # Bars (right axis)
    ax2 = ax1.twinx()
    ax2.bar(x, Acc, width=5, alpha=0.3, label="Bit Acc. (%)")
    ax2.set_ylabel("Bit Acc. (%)")
    ax2.set_ylim(0, 100)

    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}/screen_fix_a{fixed_angle}_vary_distance.png", dpi=300)
    plt.show()


def plot_angle_fixed_distance(data_rows, fixed_distance=40, save_prefix="results/plots"):
    """
    Fixed distance, varying angle
    Left y-axis: R^r, P^r, F1^r as lines (0~1)
    Right y-axis: Acc (%) as bars (0~100)
    """
    os.makedirs(save_prefix, exist_ok=True)

    rows = [r for r in data_rows if r["distance"] == fixed_distance]
    rows.sort(key=lambda x: x["angle"])

    x = [r["angle"] for r in rows]
    R = [r["R"] for r in rows]
    P = [r["P"] for r in rows]
    F1 = [r["F1"] for r in rows]
    Acc = [r["Acc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Lines (left axis)
    ax1.plot(x, R, marker="o", linewidth=2, label=r"$R^{r}$")
    ax1.plot(x, P, marker="o", linewidth=2, label=r"$P^{r}$")
    ax1.plot(x, F1, marker="o", linewidth=2, label=r"$F_{1}^{r}$")
    ax1.set_xlabel("Angle (degree)")
    ax1.set_ylabel(r"$R^{r},\ P^{r},\ F_{1}^{r}$")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    # Bars (right axis)
    ax2 = ax1.twinx()
    ax2.bar(x, Acc, width=10, alpha=0.3, label="Bit Acc. (%)")
    ax2.set_ylabel("Bit Acc. (%)")
    ax2.set_ylim(0, 100)

    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}/screen_fix_d{fixed_distance}_vary_angle.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    data = [
        {"distance": 30,  "angle": 0,   "R": 0.6709, "P": 0.8640, "F1": 0.7553, "Acc": 100.00},
        {"distance": 40,  "angle": 0,   "R": 0.3310, "P": 0.6986, "F1": 0.4492, "Acc": 100.00},
        {"distance": 50,  "angle": 0,   "R": 0.2594, "P": 0.6578, "F1": 0.3721, "Acc": 100.00},
        {"distance": 60,  "angle": 0,   "R": 0.2471, "P": 0.6477, "F1": 0.3577, "Acc": 100.00},
        {"distance": 70,  "angle": 0,   "R": 0.4590, "P": 0.7369, "F1": 0.5657, "Acc": 99.06},
        {"distance": 80,  "angle": 0,   "R": 0.6735, "P": 0.7798, "F1": 0.7228, "Acc": 98.63},
        {"distance": 90,  "angle": 0,   "R": 0.8335, "P": 0.8378, "F1": 0.8356, "Acc": 97.66},
        {"distance": 100, "angle": 0,   "R": 0.8636, "P": 0.8181, "F1": 0.8402, "Acc": 96.56},

        {"distance": 40,  "angle": -20, "R": 0.3367, "P": 0.7108, "F1": 0.4570, "Acc": 98.44},
        {"distance": 40,  "angle": -40, "R": 0.4283, "P": 0.7155, "F1": 0.5358, "Acc": 96.52},
        {"distance": 40,  "angle": -60, "R": 0.5269, "P": 0.7921, "F1": 0.6328, "Acc": 94.22},
        {"distance": 40,  "angle": 20,  "R": 0.3541, "P": 0.7819, "F1": 0.4874, "Acc": 99.53},
        {"distance": 40,  "angle": 40,  "R": 0.4629, "P": 0.7643, "F1": 0.5766, "Acc": 98.28},
        {"distance": 40,  "angle": 60,  "R": 0.5538, "P": 0.8262, "F1": 0.6631, "Acc": 96.41},
    ]

    # save_prefix fixed to 'results/plots'
    plot_distance_fixed_angle(data, fixed_angle=0, save_prefix="results/plots")
    plot_angle_fixed_distance(data, fixed_distance=40, save_prefix="results/plots")
