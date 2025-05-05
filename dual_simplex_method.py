import numpy as np


def print_table(table, basis, cb, z, f_val, iteration):
    print(f"\n--- Симплекс таблиця {iteration} ---")
    header = ["Базис", "Cб"] + \
        [f"x{i+1}" for i in range(table.shape[1] - 1)] + ["b"]
    print(" | ".join(f"{h:>7}" for h in header))
    print("-" * (9 * len(header)))
    for i, row in enumerate(table[:-1]):
        basis_var = f"x{basis[i] + 1}"
        cb_val = cb[i]
        row_str = [f"{basis_var}", f"{cb_val:.2f}"] + \
            [f"{val:.2f}" for val in row]
        print(" | ".join(f"{cell:>7}" for cell in row_str))
    z_row = ["f", " "] + [f"{zi:.2f}" for zi in z] + [f"{f_val:.2f}"]
    print(" | ".join(f"{cell:>7}" for cell in z_row))


def dual_simplex_method(A, b, c, print_steps=True):
    m, n = A.shape
    A = A.astype(float)
    b = b.astype(float)
    c = c.astype(float)
    A = np.hstack([A, np.eye(m)])
    if len(c) < n + m:
        c = np.concatenate([c, np.zeros(n + m - len(c))])
    basis = list(range(n, n + m))
    cb = np.zeros(m)
    for i, idx in enumerate(basis):
        if idx < len(c):
            cb[i] = c[idx]
    table = np.zeros((m + 1, n + m + 1))
    table[:-1, :-1] = A
    table[:-1, -1] = b
    z = cb @ A - c[:n + m]
    table[-1, :-1] = z
    table[-1, -1] = cb @ b
    iteration = 1

    if print_steps:
        print_table(table, basis, cb, z, table[-1, -1], iteration)

    while True:
        B = A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("Матриця B вироджена. Розв’язок неможливий.")
            return None, None
        xb = B_inv @ b
        table[:-1, -1] = xb
        if np.all(xb >= 0):
            break

        leaving = np.argmin(xb)
        row = B_inv[leaving]
        delta = row @ A
        sigma = cb @ B_inv @ A - c[:n + m]

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(delta < 0, -sigma / delta, np.inf)

        entering = np.argmin(ratios)
        if ratios[entering] == np.inf:
            print("Задача не має розв’язку.")
            return None, None

        pivot_col = A[:, entering]
        B[:, leaving] = pivot_col
        basis[leaving] = entering
        cb = np.zeros(m)
        for i, idx in enumerate(basis):
            if idx < len(c):
                cb[i] = c[idx]

        B_inv = np.linalg.inv(B)
        A_b = B_inv @ A
        b_new = B_inv @ b

        table[:-1, :-1] = A_b
        table[:-1, -1] = b_new
        z = cb @ A_b - c[:n + m]
        table[-1, :-1] = z
        table[-1, -1] = cb @ b_new
        if print_steps:
            print_table(table, basis, cb, z, table[-1, -1], iteration + 1)
        iteration += 1

    x = np.zeros(n)
    for i, idx in enumerate(basis):
        if idx < n:
            x[idx] = xb[i]
    optimal_value = cb @ xb
    return x, optimal_value, basis, xb


def main():
    print("=== Двоїстий симплекс-метод ===")

    m = int(input("Введіть кількість обмежень: "))
    n = int(input("Введіть кількість змінних: "))

    A = []
    b = []
    signs = []

    print("Введіть обмеження у форматі: a1 a2 ... an знак b (наприклад: 1 2 <= 10)")
    for i in range(m):
        parts = input(f"Обмеження {i+1}: ").split()
        if len(parts) != n + 2:
            raise ValueError("Неправильний формат")
        coeffs = list(map(float, parts[:n]))
        sign = parts[n]
        rhs = float(parts[n+1])
        if sign == "<=":
            A.append(coeffs)
            b.append(rhs)
        elif sign == ">=":
            A.append([-x for x in coeffs])
            b.append(-rhs)
        elif sign == "=":
            A.append(coeffs)
            b.append(rhs)
        else:
            raise ValueError("Допустимі лише <=, >=, =")
        signs.append(sign)

    c = list(map(float, input("Введіть коефіцієнти цільової функції: ").split()))
    if len(c) != n:
        raise ValueError("Невірна кількість коефіцієнтів цільової функції")

    A = np.array(A)
    b = np.array(b)
    c = np.array(c)

    c_extended = np.concatenate([c, np.zeros(m)])
    x_opt, f_opt, basis, xb = dual_simplex_method(
        A, b, c_extended, print_steps=True)
    if x_opt is not None:
        print("\n=== Результат ===")
        print("Оптимальний план:", x_opt)
        print("Значення цільової функції:", f_opt)


if __name__ == "__main__":
    main()
