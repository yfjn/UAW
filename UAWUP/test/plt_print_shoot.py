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
    os.makedirs(save_prefix, exist_ok=True)

    rows = [r for r in data_rows if r["angle"] == fixed_angle]
    rows.sort(key=lambda x: x["distance"])

    x = [r["distance"] for r in rows]
    R = [r["R"] for r in rows]
    P = [r["P"] for r in rows]
    F1 = [r["F1"] for r in rows]
    Acc = [r["Acc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(x, R, marker="o", linewidth=2, label=r"$R^{r}$")
    ax1.plot(x, P, marker="o", linewidth=2, label=r"$P^{r}$")
    ax1.plot(x, F1, marker="o", linewidth=2, label=r"$F_{1}^{r}$")
    ax1.set_xlabel("Distance (cm)")
    ax1.set_ylabel(r"$R^{r},\ P^{r},\ F_{1}^{r}$")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    ax2 = ax1.twinx()
    ax2.bar(x, Acc, width=5, alpha=0.3, label="Bit Acc. (%)")
    ax2.set_ylabel("Bit Acc. (%)")
    ax2.set_ylim(0, 100)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}/print_fix_a{fixed_angle}_vary_distance.png", dpi=300)
    plt.show()


def plot_angle_fixed_distance(data_rows, fixed_distance=40, save_prefix="results/plots"):
    os.makedirs(save_prefix, exist_ok=True)

    rows = [r for r in data_rows if r["distance"] == fixed_distance]
    rows.sort(key=lambda x: x["angle"])

    x = [r["angle"] for r in rows]
    R = [r["R"] for r in rows]
    P = [r["P"] for r in rows]
    F1 = [r["F1"] for r in rows]
    Acc = [r["Acc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(x, R, marker="o", linewidth=2, label=r"$R^{r}$")
    ax1.plot(x, P, marker="o", linewidth=2, label=r"$P^{r}$")
    ax1.plot(x, F1, marker="o", linewidth=2, label=r"$F_{1}^{r}$")
    ax1.set_xlabel("Angle (degree)")
    ax1.set_ylabel(r"$R^{r},\ P^{r},\ F_{1}^{r}$")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    ax2 = ax1.twinx()
    ax2.bar(x, Acc, width=10, alpha=0.3, label="Bit Acc. (%)")
    ax2.set_ylabel("Bit Acc. (%)")
    ax2.set_ylim(0, 100)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}/print_fix_d{fixed_distance}_vary_angle.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Print-capture scenario
    data_print = [
        {"distance": 30,  "angle": 0,   "R": 0.2574, "P": 0.6722, "F1": 0.3722, "Acc": 99.75},
        {"distance": 40,  "angle": 0,   "R": 0.0627, "P": 0.5506, "F1": 0.1126, "Acc": 100.00},
        {"distance": 50,  "angle": 0,   "R": 0.1953, "P": 0.6199, "F1": 0.2970, "Acc": 99.69},
        {"distance": 60,  "angle": 0,   "R": 0.2465, "P": 0.6570, "F1": 0.3585, "Acc": 99.66},
        {"distance": 70,  "angle": 0,   "R": 0.4545, "P": 0.7734, "F1": 0.5726, "Acc": 98.97},
        {"distance": 80,  "angle": 0,   "R": 0.6447, "P": 0.7740, "F1": 0.7035, "Acc": 98.38},
        {"distance": 90,  "angle": 0,   "R": 0.9085, "P": 0.9056, "F1": 0.9070, "Acc": 98.21},
        {"distance": 100, "angle": 0,   "R": 0.9117, "P": 0.9193, "F1": 0.9155, "Acc": 98.75},

        {"distance": 40,  "angle": -20, "R": 0.2420, "P": 0.6726, "F1": 0.3559, "Acc": 97.03},
        {"distance": 40,  "angle": -40, "R": 0.2823, "P": 0.6858, "F1": 0.4000, "Acc": 88.59},
        {"distance": 40,  "angle": -60, "R": 0.1620, "P": 0.5816, "F1": 0.2534, "Acc": 78.12},
        {"distance": 40,  "angle": 20,  "R": 0.2542, "P": 0.6052, "F1": 0.3580, "Acc": 97.66},
        {"distance": 40,  "angle": 40,  "R": 0.2760, "P": 0.6477, "F1": 0.3871, "Acc": 93.59},
        {"distance": 40,  "angle": 60,  "R": 0.1677, "P": 0.5659, "F1": 0.2588, "Acc": 77.81},
    ]

    # Save to: results/plots/...
    plot_distance_fixed_angle(data_print, fixed_angle=0, save_prefix="results/plots")
    plot_angle_fixed_distance(data_print, fixed_distance=40, save_prefix="results/plots")
