import pandas as pd
import matplotlib.pyplot as plt

PROCESSES = [1, 2, 4]
FILES = [f"./results/matvec{i}.csv" for i in PROCESSES]


# Функция для анализа данных и построения графиков
def main() -> None:
    # Загружаем данные из файлов
    dataframes = [pd.read_csv(file) for file in FILES]

    # Добавляем информацию о числе процессов (N)
    for i, df in enumerate(dataframes):
        df['processes'] = PROCESSES[i]

    # Объединяем данные в один DataFrame
    data = pd.concat(dataframes, ignore_index=True)

    # Добавляем размер матрицы как отдельную колонку
    data['size'] = data['rows'] * data['cols']

    # Группировка данных для анализа
    agg_data = data.groupby(['size', 'partition', 'processes']).mean().reset_index()

    # 1. График времени выполнения
    plt.figure(figsize=(12, 6))
    for partition in agg_data['partition'].unique():
        subset = agg_data[agg_data['partition'] == partition]
        for t in PROCESSES:
            subsubset = subset[subset['processes'] == t]
            plt.plot(subsubset['size'], subsubset['time'], label=f"{partition}, {t} процесс(ов)")
    plt.title("Время выполнения алгоритмов")
    plt.xlabel("Размер матрицы (rows * cols)")
    plt.ylabel("Время выполнения (с)")
    plt.legend()
    plt.grid()
    plt.savefig(f"time.png")
    plt.show()

    # 2. График ускорения
    seq_times = agg_data[agg_data['processes'] == 1][['size', 'partition', 'time']].set_index(['size', 'partition'])
    agg_data['speedup'] = agg_data.apply(
        lambda row: seq_times.loc[(row['size'], row['partition']), 'time'] / row['time'], axis=1
    )
    plt.figure(figsize=(12, 6))
    for partition in agg_data['partition'].unique():
        subset = agg_data[agg_data['partition'] == partition]
        for t in PROCESSES[1:]:
            subsubset = subset[subset['processes'] == t]
            plt.plot(subsubset['size'], subsubset['speedup'], label=f"{partition}, {t} процесс(ов)")
    plt.title("Ускорение алгоритмов")
    plt.xlabel("Размер матрицы (rows * cols)")
    plt.ylabel("Ускорение")
    plt.legend()
    plt.grid()
    plt.savefig(f"speedup.png")
    plt.show()

    # 3. График эффективности
    agg_data['efficiency'] = agg_data['speedup'] / agg_data['processes']
    plt.figure(figsize=(12, 6))
    for partition in agg_data['partition'].unique():
        subset = agg_data[agg_data['partition'] == partition]
        for t in PROCESSES[1:]:
            subsubset = subset[subset['processes'] == t]
            plt.plot(subsubset['size'], subsubset['efficiency'], label=f"{partition}, {t} процесс(ов)")
    plt.title("Эффективность алгоритмов")
    plt.xlabel("Размер матрицы (rows * cols)")
    plt.ylabel("Эффективность")
    plt.legend()
    plt.grid()
    plt.savefig(f"efficiency.png")
    plt.show()


if __name__ == '__main__':
    main()
