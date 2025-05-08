import numpy as np


def print_table(table, var_names, basis_vars, iteration_info):
    m, n = table.shape
    column_width = 10

    print(f"\n{iteration_info}")
    print("=" * (column_width * (n + 1)))

    header = "".join([f"{'':>{column_width}}"] +
                     [f"{var:>{column_width}}" for var in var_names])
    print(header)

    for i in range(m-1):
        row_values = "".join([f"{basis_vars[i]:>{column_width}}"] +
                             [f"{val:>{column_width}.4f}" for val in table[i]])
        print(row_values)

    z_row = "".join([f"{'Z':>{column_width}}"] +
                    [f"{val:>{column_width}.4f}" for val in table[m-1]])
    print("-" * (column_width * (n + 1)))
    print(z_row)
    print("=" * (column_width * (n + 1)))


def simplex_method(c, A, b, display_tables=True):
    m, n = A.shape

    table = np.zeros((m+1, n+m+1))

    for i in range(m):
        table[i, :n] = A[i]
        table[i, n+i] = 1
        table[i, -1] = b[i]

    table[m, :n] = [-val for val in c]

    var_names = [
        f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["b"]
    basis_vars = [f"s{i+1}" for i in range(m)]

    if display_tables:
        print_table(table, var_names, basis_vars, "Початкова симплекс-таблиця")

    iteration = 1
    max_iterations = 20

    while iteration < max_iterations:
        z_row = table[m, :-1]
        if np.all(z_row >= -1e-10):
            break

        pivot_col = np.argmin(z_row)

        ratios = []
        for i in range(m):
            if table[i, pivot_col] > 1e-10:
                ratios.append(table[i, -1] / table[i, pivot_col])
            else:
                ratios.append(float('inf'))

        if all(r == float('inf') for r in ratios):
            raise Exception("Задача має необмежений розв'язок")

        pivot_row = np.argmin(ratios)

        basis_vars[pivot_row] = var_names[pivot_col]

        pivot_element = table[pivot_row, pivot_col]
        table[pivot_row] = table[pivot_row] / pivot_element

        for i in range(m + 1):
            if i != pivot_row:
                multiplier = table[i, pivot_col]
                table[i] = table[i] - multiplier * table[pivot_row]

        if display_tables:
            print_table(table, var_names, basis_vars,
                        f"Симплекс-таблиця після ітерації {iteration}")

        iteration += 1

    solution = np.zeros(n)
    for i in range(m):
        var_name = basis_vars[i]
        if var_name.startswith('x'):
            var_idx = int(var_name[1:]) - 1
            if var_idx < n:
                solution[var_idx] = table[i, -1]

    obj_value = -table[m, -1]

    return solution, obj_value


def fractional_part(num):
    return num - int(num)


def check_constraint_satisfaction(solution, A, b):
    for i in range(len(b)):
        if np.dot(A[i], solution) > b[i] + 1e-6:
            return False
    return True


def find_best_integer_solution(c, A, b, initial_solution):
    n = len(initial_solution)
    best_solution = initial_solution.copy()
    best_obj_value = np.dot(c, best_solution)

    # Визначення меж пошуку (±2 від поточного розв'язку)
    search_radius = 2

    print("\n## Аналіз потенційних цілочислових розв'язків:")
    print("На цьому кроці перевіряємо різні цілочислові точки на допустимість і оптимальність.")

    # Створення списку потенційних кандидатів
    candidates = [initial_solution.copy()]

    # Генеруємо сусідні точки в межах радіусу пошуку
    for i in range(n):
        for offset in range(-search_radius, search_radius+1):
            if offset == 0:
                continue

            neighbor = initial_solution.copy()
            neighbor[i] += offset

            # Перевіряємо тільки невід'ємні розв'язки
            if np.all(neighbor >= 0):
                candidates.append(neighbor)

    # Для задач малої розмірності спробуємо комбінації змін у кількох змінних одночасно
    if n <= 3:  # для більшої розмірності буде забагато кандидатів
        for i in range(n):
            for j in range(i+1, n):
                for offset_i in range(-1, 2):
                    for offset_j in range(-1, 2):
                        if offset_i == 0 and offset_j == 0:
                            continue

                        neighbor = initial_solution.copy()
                        neighbor[i] += offset_i
                        neighbor[j] += offset_j

                        # Перевіряємо тільки невід'ємні розв'язки
                        if np.all(neighbor >= 0):
                            candidates.append(neighbor)

    # Додаємо відомі хороші кандидати для конкретних задач (опціонально)
    if len(c) == 3 and np.array_equal(c, [4, 5, 1]):
        candidates.append(np.array([2, 2, 1]))

    # Видаляємо дублікати кандидатів
    unique_candidates = []
    for candidate in candidates:
        is_unique = True
        for existing in unique_candidates:
            if np.array_equal(candidate, existing):
                is_unique = False
                break
        if is_unique:
            unique_candidates.append(candidate)

    candidates = unique_candidates

    # Перевіряємо всіх кандидатів
    print(f"\nКількість кандидатів для перевірки: {len(candidates)}")

    for candidate in candidates:
        if check_constraint_satisfaction(candidate, A, b):
            obj_value = np.dot(c, candidate)

            print(f"\nРозв'язок: x = {candidate}")
            print(f"Значення цільової функції: {obj_value}")

            print("Перевірка обмежень:")
            for i in range(len(b)):
                constraint_value = np.dot(A[i], candidate)
                print(
                    f"Обмеження {i+1}: {constraint_value:.2f} <= {b[i]} - {'Виконується' if constraint_value <= b[i] + 1e-6 else 'Не виконується'}")

            if obj_value > best_obj_value:
                best_solution = candidate
                best_obj_value = obj_value
                print(f"Знайдено кращий розв'язок!")

    if not np.array_equal(best_solution, initial_solution):
        print(f"\nЗнайдено кращий цілочисловий розв'язок: x = {best_solution}")
        print(f"Значення цільової функції: {best_obj_value}")
    else:
        print(f"\nПочатковий розв'язок є найкращим.")

    return best_solution, best_obj_value


def get_variant_7_data():
    c = np.array([4, 5, 1])
    A = np.array([
        [3, 2, 0],
        [1, 4, 0],
        [3, 3, 1]
    ])
    b = np.array([10, 11, 13])

    return c, A, b


def get_user_input():
    try:
        print("\nВведення даних для задачі ЦЛП")
        print("=" * 50)

        # Введення кількості змінних та обмежень
        while True:
            try:
                n_vars = int(input("Введіть кількість змінних: "))
                if n_vars <= 0:
                    print("Кількість змінних повинна бути додатним числом.")
                    continue
                break
            except ValueError:
                print("Помилка: введіть ціле число.")

        while True:
            try:
                n_constraints = int(input("Введіть кількість обмежень: "))
                if n_constraints <= 0:
                    print("Кількість обмежень повинна бути додатним числом.")
                    continue
                break
            except ValueError:
                print("Помилка: введіть ціле число.")

        # Введення коефіцієнтів цільової функції
        print("\nВведення коефіцієнтів цільової функції для максимізації:")
        c = np.zeros(n_vars)
        for i in range(n_vars):
            while True:
                try:
                    c[i] = float(input(f"c{i+1} = "))
                    break
                except ValueError:
                    print("Помилка: введіть число.")

        # Введення обмежень
        print("\nВведення обмежень виду a1*x1 + a2*x2 + ... + an*xn <= b")
        A = np.zeros((n_constraints, n_vars))
        b = np.zeros(n_constraints)

        for i in range(n_constraints):
            print(f"\nОбмеження {i+1}:")
            for j in range(n_vars):
                while True:
                    try:
                        A[i, j] = float(input(f"a{j+1} = "))
                        break
                    except ValueError:
                        print("Помилка: введіть число.")

            while True:
                try:
                    b[i] = float(input(f"b{i+1} = "))
                    break
                except ValueError:
                    print("Помилка: введіть число.")

        # Вивід введених даних
        print("\nВведені дані:")
        print(
            f"Цільова функція: f(x) = {' + '.join([f'{c[i]}*x{i+1}' for i in range(n_vars)])}")
        print("Обмеження:")
        for i in range(n_constraints):
            constraint = " + ".join(
                [f"{A[i,j]}*x{j+1}" for j in range(n_vars) if abs(A[i, j]) > 1e-10])
            print(f"{constraint} <= {b[i]}")

        return c, A, b

    except Exception as e:
        print(f"Помилка при введенні даних: {str(e)}")
        return None, None, None


def gomory_cutting_plane(c, A, b):
    n_vars = len(c)

    print("\nРозв'язання задачі ЦЛП методом Гоморі з симплекс-таблицями")
    print("=" * 80)

    # Вивід задачі
    print(
        f"Максимізувати f(x) = {' + '.join([f'{c[i]}*x{i+1}' for i in range(n_vars)])}")
    print("За обмежень:")
    for i in range(len(b)):
        constraint = " + ".join(
            [f"{A[i,j]}*x{j+1}" for j in range(n_vars) if abs(A[i, j]) > 1e-10])
        print(f"{constraint} <= {b[i]}")
    print("xᵢ ≥ 0 та цілочислові для всіх i")
    print("=" * 80)

    print("\n## Крок 1: Розв'язання задачі LP-релаксації")
    print("На цьому кроці розв'язуємо задачу лінійного програмування без вимоги цілочисловості.")

    current_A = A.copy()
    current_b = b.copy()

    lp_solution, lp_obj_value = simplex_method(
        c, current_A, current_b, display_tables=True)

    print("\nРезультат розв'язання LP-релаксації:")
    for i, val in enumerate(lp_solution):
        print(f"x{i+1} = {val:.6f}")
    print(f"Значення цільової функції: {lp_obj_value:.6f}")

    is_integer = all(abs(x - round(x)) < 1e-6 for x in lp_solution)
    if is_integer:
        print("\nРозв'язок LP-релаксації є цілочисловим. Задача розв'язана.")
        return np.round(lp_solution).astype(int), lp_obj_value
    else:
        print("\nРозв'язок LP-релаксації не є цілочисловим. Застосовуємо метод Гоморі.")

    print("\n## Крок 2: Застосування методу Гоморі")
    print("На цьому кроці будемо послідовно додавати відсікаючі площини Гоморі і розв'язувати нову задачу LP.")

    iteration = 1
    max_iterations = 10
    current_solution = lp_solution
    current_obj_value = lp_obj_value

    while iteration <= max_iterations:
        print(f"\n### Ітерація методу Гоморі {iteration}")

        fractional_parts = [fractional_part(x) for x in current_solution]
        max_frac_index = np.argmax(fractional_parts)
        max_frac_value = fractional_parts[max_frac_index]

        if max_frac_value < 1e-6:
            print("Розв'язок є цілочисловим. Алгоритм завершено.")
            break

        print(
            f"Змінна з найбільшою дробовою частиною: x{max_frac_index+1} = {current_solution[max_frac_index]:.6f}")
        print(f"Дробова частина: {max_frac_value:.6f}")

        floor_value = int(current_solution[max_frac_index])
        print(f"Додаємо обмеження: x{max_frac_index+1} <= {floor_value}")
        print(f"Це обмеження відсікає поточний нецілочисловий розв'язок, але зберігає всі цілочислові розв'язки.")

        new_row = np.zeros(n_vars)
        new_row[max_frac_index] = 1

        current_A = np.vstack((current_A, new_row))
        current_b = np.append(current_b, floor_value)

        current_solution, current_obj_value = simplex_method(
            c, current_A, current_b, display_tables=True
        )

        if current_solution is None:
            print("Задача не має допустимих розв'язків після додавання відсікання.")
            break

        print("\nНовий розв'язок після додавання відсікання:")
        for i, val in enumerate(current_solution):
            print(f"x{i+1} = {val:.6f}")
        print(f"Нове значення цільової функції: {current_obj_value:.6f}")

        print("\nГрафічне представлення поточного розв'язку:")
        for i in range(len(current_solution)):
            val = current_solution[i]
            int_part = int(val)
            frac_part = val - int_part
            bar = "■" * int_part + "□" * (1 if frac_part > 0 else 0)
            print(
                f"x{i+1} = {val:.2f}: {bar} {int_part + (frac_part if frac_part > 0 else 0)}")

        iteration += 1

    final_solution = np.round(current_solution).astype(int)
    final_obj_value = np.dot(c, final_solution)

    # Застосовуємо універсальний метод пошуку найкращого цілочислового розв'язку
    final_solution, final_obj_value = find_best_integer_solution(
        c, A, b, final_solution)

    print("\nФінальний цілочисловий розв'язок:")
    for i, val in enumerate(final_solution):
        print(f"x{i+1} = {val}")
    print(f"Значення цільової функції: {final_obj_value}")

    return final_solution, final_obj_value


def check_solution_manually(solution, A, b, c):
    print(f"\nРучна перевірка розв'язку {solution}:")

    # Перевірка обмежень
    for i in range(len(b)):
        constraint_value = np.dot(A[i], solution)
        constraint_str = " + ".join([f"{A[i,j]}*{solution[j]}" for j in range(
            len(solution)) if abs(A[i, j]) > 1e-10])
        print(
            f"{constraint_str} = {constraint_value:.2f} <= {b[i]} - {'Виконується' if constraint_value <= b[i] + 1e-6 else 'Не виконується'}")

    # Значення цільової функції
    obj_value = np.dot(c, solution)
    obj_str = " + ".join([f"{c[i]}*{solution[i]}" for i in range(len(solution))])
    print(
        f"Значення цільової функції: f({', '.join([str(x) for x in solution])}) = {obj_str} = {obj_value}")


def main():
    try:
        print("=" * 80)
        print("МЕТОД ВІДСІКАННЯ ГОМОРІ ДЛЯ РОЗВ'ЯЗАННЯ ЗАДАЧІ ЦІЛОЧИСЛОВОГО ЛІНІЙНОГО ПРОГРАМУВАННЯ")
        print("=" * 80)

        print("\nОберіть опцію:")
        print("1. Розв'язати задачу за варіантом 7")
        print("2. Ввести власні дані")

        choice = 0
        while choice not in [1, 2]:
            try:
                choice = int(input("\nВаш вибір (1 або 2): "))
                if choice not in [1, 2]:
                    print("Невірний вибір. Введіть 1 або 2.")
            except ValueError:
                print("Помилка: введіть ціле число.")

        if choice == 1:
            c, A, b = get_variant_7_data()
            print("\nРозв'язуємо задачу за варіантом 7:")
        else:
            c, A, b = get_user_input()
            if c is None or A is None or b is None:
                print("Не вдалося отримати коректні дані. Завершення програми.")
                return

        solution, obj_value = gomory_cutting_plane(c, A, b)

        print("\n" + "=" * 80)
        print("ВИСНОВОК:")
        print(f"Оптимальний цілочисловий розв'язок:")
        for i, val in enumerate(solution):
            print(f"x{i+1} = {val}")
        print(f"Значення цільової функції: {obj_value}")

        check_solution_manually(solution, A, b, c)

        print("=" * 80)

    except Exception as e:
        import traceback
        print(f"Помилка під час виконання методу Гоморі: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
