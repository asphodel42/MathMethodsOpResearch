import numpy as np
import matplotlib.pyplot as plt


def plot_example_1():
    """
    Example 1: F = x1 * x2, constraint 2x1 + x2 = 8
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Побудова поверхні F(x1,x2) = x1*x2
    x1 = np.linspace(-8, 8, 300)
    x2 = np.linspace(-8, 8, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = X1 * X2
    surf = ax.plot_surface(X1, X2, Z, cmap='jet', alpha=0.7, edgecolor='none')

    # Побудова площини обмеження 2x1 + x2 = 8
    X1p = np.linspace(-8, 8, 50)
    Zp = np.linspace(np.min(Z), np.max(Z), 50)
    X1p, Zp = np.meshgrid(X1p, Zp)
    X2p = 8 - 2 * X1p
    ax.plot_surface(X1p, X2p, Zp, color='black', alpha=0.5, edgecolor='none')

    # Лінія перетину surface ∩ plane
    t = np.linspace(-4, 4, 200)
    x1_line = t
    x2_line = 8 - 2*t
    z_line = x1_line * x2_line
    ax.plot(x1_line, x2_line, z_line, color='k', linewidth=2)

    ax.set_title('Example 1: $F = x_1 x_2$, $2x_1 + x_2 = 8$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$F$')
    plt.colorbar(surf, shrink=0.5, aspect=10)


def plot_example_2():
    """
    Example 2: F = x1 + x2, constraint x1^2 + x2^2 = 1
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Побудова циліндра x1^2 + x2^2 = 1
    theta = np.linspace(0, 2*np.pi, 200)
    z_vals = np.linspace(-4, 4, 200)
    Θ, Zc = np.meshgrid(theta, z_vals)
    Xc = np.cos(Θ)
    Yc = np.sin(Θ)
    ax.plot_surface(Xc, Yc, Zc, cmap='jet', alpha=0.8, edgecolor='none')

    # Побудова площини F = x1 + x2
    xp = np.linspace(-1.5, 1.5, 200)
    yp = np.linspace(-1.5, 1.5, 200)
    Xp, Yp = np.meshgrid(xp, yp)
    Zp = Xp + Yp
    ax.plot_surface(Xp, Yp, Zp, cmap='viridis', alpha=0.6, edgecolor='none')

    # Лінія перетину
    theta_line = np.linspace(0, 2*np.pi, 300)
    x1_line = np.cos(theta_line)
    x2_line = np.sin(theta_line)
    z_line = x1_line + x2_line
    ax.plot(x1_line, x2_line, z_line, color='k', linewidth=2)

    ax.set_title('Example 2: $F = x_1 + x_2$, $x_1^2 + x_2^2=1$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$F$')


def plot_example_3():
    """
    Example 3: f = x1^2 + x2^2, constraint 2x1 + x2 = 2
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Побудова параболоїда f = x1^2 + x2^2
    x1 = np.linspace(-1, 2, 200)
    x2 = np.linspace(-2, 3, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = X1**2 + X2**2
    surf = ax.plot_surface(X1, X2, Z, cmap='coolwarm',
                           alpha=0.7, edgecolor='none')

    # Побудова площини обмеження 2x1 + x2 = 2
    X1p = np.linspace(-1, 2, 50)
    Zp = np.linspace(np.min(Z), np.max(Z), 50)
    X1p, Zp = np.meshgrid(X1p, Zp)
    X2p = 2 - 2 * X1p
    ax.plot_surface(X1p, X2p, Zp, color='black', alpha=0.5, edgecolor='none')

    # Лінія перетину
    t = np.linspace(-1, 2, 200)
    x1_line = t
    x2_line = 2 - 2*t
    z_line = x1_line**2 + x2_line**2
    ax.plot(x1_line, x2_line, z_line, color='k', linewidth=2)

    ax.set_title('Example 3: $f = x_1^2 + x_2^2$, $2x_1 + x_2=2$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f$')
    plt.colorbar(surf, shrink=0.5, aspect=10)


def main():
    plot_example_1()
    plot_example_2()
    plot_example_3()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
