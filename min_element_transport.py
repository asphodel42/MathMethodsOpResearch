import pandas as pd
import csv


def print_cost_table(costs, supply, demand):
    df = pd.DataFrame(costs,
                      index=[f"Підпр.{i+1}" for i in range(len(costs))],
                      columns=[f"Спож.{j+1}" for j in range(len(costs[0]))])
    df["Залишок"] = supply

    demand_row = [f"{d}" for d in demand] + [""]
    df.loc["Потреба"] = demand_row
    print("=== Початкова таблиця вартостей перевезення ===")
    print(df)


def print_step_table(costs, allocation, step, action, supply, demand):
    print(f"\n--- Крок {step} ---")
    print(f"{action}")

    df = pd.DataFrame(index=[f"Підпр.{i+1}" for i in range(len(allocation))],
                      columns=[f"Спож.{j+1}" for j in range(len(allocation[0]))])

    for i in range(len(allocation)):
        for j in range(len(allocation[0])):
            if allocation[i][j] != 0:
                df.iloc[i, j] = f"{allocation[i][j]}({costs[i][j]})"
            else:
                df.iloc[i, j] = ""

    df["Залишок"] = supply

    demand_row = [f"{d}" for d in demand] + [""]
    df.loc["Потреба"] = demand_row

    print(df.fillna(""))


def save_to_csv(allocation, costs, filename="allocation_result.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([""] + [f"Спож.{j+1}" for j in range(len(costs[0]))])
        for i in range(len(allocation)):
            row = [f"Підпр.{i+1}"]
            for j in range(len(allocation[0])):
                if allocation[i][j] != 0:
                    row.append(f"{allocation[i][j]}({costs[i][j]})")
                else:
                    row.append("")
            writer.writerow(row)


def min_element_method(supply, demand, costs):
    supply = supply.copy()
    demand = demand.copy()
    allocation = [[0] * len(demand) for _ in range(len(supply))]

    print_cost_table(costs, supply, demand)
    step = 1
    while any(s > 0 for s in supply) and any(d > 0 for d in demand):
        min_val = float('inf')
        min_cell = (-1, -1)
        for i in range(len(supply)):
            for j in range(len(demand)):
                if supply[i] > 0 and demand[j] > 0 and costs[i][j] < min_val:
                    min_val = costs[i][j]
                    min_cell = (i, j)

        i, j = min_cell
        qty = min(supply[i], demand[j])
        allocation[i][j] = qty
        supply[i] -= qty
        demand[j] -= qty
        action = f"Вибрана клітинка ({i+1}, {j+1}) з мін. вартістю {costs[i][j]}, розміщено {qty} одиниць"
        print_step_table(costs, allocation, step, action, supply, demand)
        step += 1

    total_cost = sum(allocation[i][j] * costs[i][j]
                     for i in range(len(supply))
                     for j in range(len(demand)))

    save_to_csv(allocation, costs)
    return allocation, total_cost


def main():
    supply = [43, 20, 30, 32]
    demand = [18, 50, 22, 35]
    costs = [
        [4, 9, 1, 3],
        [2, 5, 5, 6],
        [2, 5, 10, 4],
        [3, 7, 2, 6]
    ]
    allocation, total_cost = min_element_method(supply, demand, costs)

    print("\n=== Фінальний розподіл ===")
    df = pd.DataFrame(allocation, index=["A", "B", "C", "D"], columns=[
                      "C1", "C2", "C3", "C4"])
    print(df)

    print(f"\n=== Загальна вартість перевезень: {total_cost} ===")


if __name__ == "__main__":
    main()
