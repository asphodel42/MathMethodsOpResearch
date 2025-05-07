import numpy as np
import pandas as pd


def display_table(allocation, cost, supply, demand, step, title="Таблиця розподілу (allocation)"):
    m, n = cost.shape
    row_labels = [f'A{i+1}' for i in range(m)]
    col_labels = [f'B{i+1}' for i in range(n)]
    df_alloc = pd.DataFrame(allocation, index=row_labels, columns=col_labels)
    df_cost = pd.DataFrame(cost, index=row_labels, columns=col_labels)
    df_alloc['Supply'] = supply
    df_cost['Supply'] = supply
    demand_row = pd.Series(demand, index=col_labels)
    df_alloc.loc['Demand'] = demand_row
    df_cost.loc['Demand'] = demand_row
    print(f"\nКрок {step}: {title}")
    print(df_alloc)
    print(f"\nКрок {step}: Таблиця вартостей (cost)")
    print(df_cost)
    print("="*60)


def display_reduced_costs(reduced_costs, step):
    m, n = reduced_costs.shape
    row_labels = [f'A{i+1}' for i in range(m)]
    col_labels = [f'B{i+1}' for i in range(n)]
    df = pd.DataFrame(reduced_costs, index=row_labels, columns=col_labels)
    print(f"\nКрок {step}: Матриця редукованих витрат (дельти)")
    print(df)
    print("="*60)


def input_data():
    try:
        m = int(input("Введіть кількість постачальників (рядків): "))
        n = int(input("Введіть кількість споживачів (стовпців): "))
        print("Введіть матрицю вартостей (по рядках, через пробіл):")
        cost = []
        for i in range(m):
            row = list(map(float, input(f"Рядок {i+1}: ").split()))
            if len(row) != n:
                print(
                    "Помилка: кількість елементів у рядку не співпадає з кількістю споживачів!")
                return None, None, None
            cost.append(row)
        cost = np.array(cost, dtype=float)
        supply = list(
            map(float, input(f"Введіть обсяги постачання ({m} чисел): ").split()))
        if len(supply) != m:
            print("Помилка: кількість постачальників не співпадає!")
            return None, None, None
        demand = list(
            map(float, input(f"Введіть потреби споживачів ({n} чисел): ").split()))
        if len(demand) != n:
            print("Помилка: кількість споживачів не співпадає!")
            return None, None, None
        supply = np.array(supply, dtype=float)
        demand = np.array(demand, dtype=float)
        if np.any(supply < 0) or np.any(demand < 0):
            print("Помилка: усі значення повинні бути невід'ємними!")
            return None, None, None
        if not np.isclose(supply.sum(), demand.sum()):
            print(
                f"Помилка: сума постачання ({supply.sum()}) не дорівнює сумі потреб ({demand.sum()})!")
            return None, None, None
        return supply, demand, cost
    except Exception as e:
        print(f"Помилка введення: {e}")
        return None, None, None


def main():
    print("Оберіть режим роботи:")
    print("1 - Готові дані (Варіант 7)")
    print("2 - Ввести дані вручну")
    mode = input("Ваш вибір (1/2): ").strip()
    if mode == '1':
        supply = np.array([43, 20, 30, 32], dtype=float)
        demand = np.array([18, 50, 22, 35], dtype=float)
        cost = np.array([
            [4, 9, 1, 3],
            [2, 5, 5, 6],
            [2, 5, 10, 4],
            [3, 7, 2, 6]
        ], dtype=float)
    elif mode == '2':
        supply, demand, cost = input_data()
        if supply is None:
            print("Введення даних некоректне. Завершення роботи.")
            return
    else:
        print("Некоректний вибір режиму!")
        return

    allocation = min_element_method(supply, demand, cost)
    print(f"\nПочатковий план, вартість: {np.sum(allocation * cost)}")
    allocation = modi_optimization(allocation, cost, supply, demand)


def min_element_method(supply, demand, cost):
    m, n = cost.shape
    allocation = np.zeros((m, n), dtype=float)
    supply_left = supply.copy()
    demand_left = demand.copy()
    cost_work = cost.copy()
    step = 1
    while supply_left.sum() > 0 and demand_left.sum() > 0:
        i, j = np.unravel_index(np.argmin(cost_work), cost_work.shape)
        x = min(supply_left[i], demand_left[j])
        allocation[i, j] = x
        supply_left[i] -= x
        demand_left[j] -= x
        display_table(allocation, cost, supply_left, demand_left, step)
        step += 1
        if supply_left[i] == 0:
            cost_work[i, :] = np.inf
        if demand_left[j] == 0:
            cost_work[:, j] = np.inf
    return allocation


def find_potentials(allocation, cost):
    m, n = allocation.shape
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0
    basis = np.argwhere(allocation > 0)
    for _ in range(m + n):
        for i, j in basis:
            if np.isnan(u[i]) and not np.isnan(v[j]):
                u[i] = cost[i, j] - v[j]
            elif not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = cost[i, j] - u[i]
    return u, v


def find_reduced_costs(allocation, cost, u, v):
    m, n = allocation.shape
    reduced_costs = np.full((m, n), np.nan)
    for i in range(m):
        for j in range(n):
            if allocation[i, j] == 0:
                reduced_costs[i, j] = cost[i, j] - u[i] - v[j]
    return reduced_costs


def find_cycle(allocation, start):
    m, n = allocation.shape
    basis = set(map(tuple, np.argwhere(allocation > 0)))
    basis.add(start)
    print(
        f"Базисні клітинки (з урахуванням нової): {[f'A{i+1}B{j+1}' for i,j in sorted(basis)]}")

    def dfs(path, hor):
        current = path[-1]
        if len(path) > 3 and current == start and len(path) % 2 == 1:
            print(f"Знайдено цикл: {[f'A{i+1}B{j+1}' for i,j in path]}")
            return path
        i, j = current
        next_steps = []
        if hor:
            for jj in range(n):
                if jj != j and (i, jj) in basis and (i, jj) not in path or ((i, jj) == start and len(path) > 3):
                    next_steps.append((i, jj))
        else:
            for ii in range(m):
                if ii != i and (ii, j) in basis and (ii, j) not in path or ((ii, j) == start and len(path) > 3):
                    next_steps.append((ii, j))
        for next_cell in next_steps:
            print(
                f"Крок: {[f'A{i+1}B{j+1}' for i,j in path]} -> A{next_cell[0]+1}B{next_cell[1]+1}")
            res = dfs(path + [next_cell], not hor)
            if res:
                return res
        return None
    cycle = dfs([start], True)
    if cycle is None:
        cycle = dfs([start], False)
    if cycle:
        print(f"Фінальний цикл: {[f'A{i+1}B{j+1}' for i,j in cycle]}")
    else:
        print("Цикл не знайдено!")
    return cycle if cycle else [start]


def modi_optimization(allocation, cost, supply, demand):
    step = 1
    while True:
        u, v = find_potentials(allocation, cost)
        reduced_costs = find_reduced_costs(allocation, cost, u, v)
        display_table(allocation, cost, supply, demand,
                      step, title="Поточний план")
        display_reduced_costs(reduced_costs, step)
        min_delta = np.nanmin(reduced_costs)
        if min_delta >= 0 or np.isnan(min_delta):
            print(
                f"\nПлан оптимальний. Оптимальна вартість: {np.sum(allocation * cost)}")
            break
        i0, j0 = np.unravel_index(np.nanargmin(
            reduced_costs), reduced_costs.shape)
        print(
            f"\nВибираємо клітину для введення у базис: A{i0+1}B{j0+1} (дельта={min_delta})")
        cycle = find_cycle(allocation, (i0, j0))
        print(f"Цикл для коригування: {[f'A{i+1}B{j+1}' for i,j in cycle]}")
        # Використовуємо цикл без останньої появи стартової клітинки
        if len(cycle) > 1 and cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        minus_cells = cycle[1::2]
        if not minus_cells:
            print(
                "Помилка: не знайдено жодної клітини для віднімання у циклі. Оптимізацію завершено.")
            break
        theta = min([allocation[i, j] for i, j in minus_cells])
        print(f"Мінімальне значення для коригування (theta): {theta}")
        # Коригуємо план (тільки унікальні клітинки циклу)
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:
                allocation[i, j] += theta
            else:
                allocation[i, j] -= theta
        # Видалення зайвої базисної клітинки (allocation==0) після коригування
        zero_cells = [(i, j) for idx, (i, j) in enumerate(cycle)
                      if allocation[i, j] == 0 and (i, j) != (i0, j0)]
        if zero_cells:
            i_del, j_del = zero_cells[0]
            allocation[i_del, j_del] = 0
            print(f"Видалено з базису клітинку: A{i_del+1}B{j_del+1}")
        print(f"План після коригування:\n{pd.DataFrame(allocation)}")
        step += 1
    return allocation


if __name__ == "__main__":
    main()
