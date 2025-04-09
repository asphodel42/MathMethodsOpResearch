import numpy as np


def print_table(A_ext, b, c_ext, basis, B_inv, x_b, delta, problem_type, iteration):
    column_width = 8
    headers = [f"x{i+1}" for i in range(len(c_ext))] + ["b"]

    print(f"\n--- Симплекс таблиця {iteration} ---")
    print("".join([f"{'':>{column_width}}"] +
          [f"{h:>{column_width}}" for h in headers]))

    for i, row_index in enumerate(basis):
        row = list(A_ext[i]) + [x_b[i]]
        row_label = f"x{row_index+1}"
        row_str = [f"{row_label:>{column_width}}"] + \
            [f"{val:>{column_width}.2f}" for val in row]
        print("".join(row_str))

    # Рядок f
    c_b = c_ext[basis]
    y = c_b @ B_inv
    adjusted_delta = [-val if problem_type == 'max' else val for val in delta]
    z_value = c_b @ x_b
    z_value = -z_value if problem_type == 'max' else z_value
    reduced_row = [f"{val:>{column_width}.2f}" for val in adjusted_delta] + \
        [f"{z_value:>{column_width}.2f}"]

    print("".join([f"{'f':>{column_width}}"] + reduced_row))


def simplex_method(A_ext, b, c_ext, basis, problem_type):
    iteration = 1
    while True:
        B = A_ext[:, basis]
        B_inv = np.linalg.inv(B)
        x_b = B_inv @ b
        c_b = c_ext[basis]
        y = c_b @ B_inv
        delta = y @ A_ext - c_ext

        print_table(A_ext, b, c_ext, basis, B_inv, x_b,
                    delta, problem_type, iteration)
        iteration += 1

        if np.all(delta <= 1e-10):
            x = np.zeros_like(c_ext)
            x[basis] = x_b
            result = x[:original_vars]
            optimal_value = c_ext[:original_vars] @ result
            if problem_type == 'max':
                optimal_value *= -1
            return result, optimal_value

        entering = np.argmax(delta)
        d = B_inv @ A_ext[:, entering]

        if np.all(d <= 1e-10):
            raise Exception("Цільова функція необмежена")

        ratios = np.array(
            [x_b[i] / d[i] if d[i] > 0 else np.inf for i in range(len(b))])
        leaving = np.argmin(ratios)
        basis[leaving] = entering


def main():
    global original_vars
    print("=== Симплекс-метод ===")
    m = int(input("Введіть кількість обмежень (рядків): "))
    n = int(input("Введіть кількість змінних (стовпців): "))
    original_vars = n  # зберігаємо для кінцевого виводу

    problem_type = input("Тип задачі (min/max): ").strip().lower()
    if problem_type not in ('min', 'max'):
        raise ValueError("Невірно вказано тип задачі. Введіть 'min' або 'max'")

    A = []
    b = []
    signs = []

    print("Введіть обмеження у форматі: a1 a2 ... an знак b (наприклад: 1 2 <= 10)")
    for i in range(m):
        parts = input(f"Обмеження {i+1}: ").split()
        if len(parts) != n + 2:
            raise ValueError(
                "Некоректний формат. Має бути n коефіцієнтів, знак, b")

        coeffs = list(map(float, parts[:n]))
        sign = parts[n]
        rhs = float(parts[n+1])

        A.append(coeffs)
        b.append(rhs)
        signs.append(sign)

    A = np.array(A)
    b = np.array(b)

    slack_columns = []
    c_slack = []
    basis = []
    slack_var_index = n  # індекси змінних починаються після основних

    for i, sign in enumerate(signs):
        slack_col = np.zeros((m,))
        if sign == "<=":
            slack_col[i] = 1
            basis.append(slack_var_index)
            c_slack.append(0)
        elif sign == ">=":
            slack_col[i] = -1
            basis.append(slack_var_index)
            c_slack.append(0)
        elif sign == "=":
            slack_col[i] = 0
            basis.append(slack_var_index)
            c_slack.append(0)
        else:
            raise ValueError("Невірний знак обмеження")

        slack_columns.append(slack_col)
        slack_var_index += 1

    # Формування повної матриці A та вектора c
    A_slack = np.column_stack(slack_columns)
    A_ext = np.hstack((A, A_slack))
    c_main = list(
        map(float, input("Введіть коефіцієнти цільової функції c: ").split()))
    if len(c_main) != n:
        raise ValueError(
            "Неправильна кількість коефіцієнтів у цільовій функції")
    c = np.array(c_main + c_slack)

    if problem_type == 'max':
        c = -c

    x_opt, f_opt = simplex_method(A_ext, b, c, basis, problem_type)
    print("\n=== Результат ===")
    print("Оптимальний план:", x_opt)
    print("Значення цільової функції:", f_opt)


if __name__ == "__main__":
    main()
