import pandas as pd
import matplotlib.pyplot as plt

# Пути к файлам
PROCESSES = [1, 2, 4]
FILES = [f"./results/cannon{i}.csv" for i in PROCESSES]


# Функция для анализа и визуализации данных
def main() -> None:
    # Загружаем данные
    dataframes = [pd.read_csv(file) for file in FILES]

    # Добавляем информацию о числе процессов
    for i, df in enumerate(dataframes):
        df['processes'] = PROCESSES[i]

    # Объединяем данные в один DataFrame
    data = pd.concat(dataframes, ignore_index=True)

    # Рассчитываем ускорение
    seq_times = data[data['processes'] == 1][['size', 'time']].set_index('size')
    data['speedup'] = data.apply(
        lambda row: seq_times.loc[row['size'], 'time'] / row['time'] if row['processes'] > 1 else 1, axis=1
    )

    # Рассчитываем эффективность
    data['efficiency'] = data['speedup'] / data['processes']

    # Построение графиков
    plt.figure(figsize=(12, 9))

    # 1. Время выполнения
    plt.subplot(3, 1, 1)
    for p in PROCESSES:
        subset = data[data['processes'] == p]
        plt.plot(subset['size'], subset['time'], label=f"{p} процесс(ов)")
    plt.title("Время выполнения алгоритма Кэннона")
    plt.xlabel("Размер матрицы (size)")
    plt.ylabel("Время выполнения (с)")
    plt.legend()
    plt.grid()

    # 2. Ускорение
    plt.subplot(3, 1, 2)
    for p in PROCESSES[1:]:
        subset = data[data['processes'] == p]
        plt.plot(subset['size'], subset['speedup'], label=f"{p} процесс(ов)")
    plt.title("Ускорение алгоритма Кэннона")
    plt.xlabel("Размер матрицы (size)")
    plt.ylabel("Ускорение")
    plt.legend()
    plt.grid()

    # 3. Эффективность
    plt.subplot(3, 1, 3)
    for p in PROCESSES[1:]:
        subset = data[data['processes'] == p]
        plt.plot(subset['size'], subset['efficiency'], label=f"{p} процесс(ов)")
    plt.title("Эффективность алгоритма Кэннона")
    plt.xlabel("Размер матрицы (size)")
    plt.ylabel("Эффективность")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"cannon.png")
    plt.show()


if __name__ == '__main__':
    main()
