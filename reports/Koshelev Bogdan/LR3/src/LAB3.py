import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def solution() -> None:
    QUALITY: str = 'quality'
    TARGET: str = 'good'

    # 1. Загрузка данных и создание бинарной целевой переменной
    wine: pd.DataFrame = pd.read_csv('winequality-white.csv', sep=';')
    wine[TARGET] = (wine[QUALITY] >= 7).astype(int)
    print("Распределение классов (0 - обычное, 1 - хорошее):")
    print(wine[TARGET].value_counts(), "\n")

    # 2. Разделение выборки и стандартизация признаков
    X_train, X_test, y_train, y_test = train_test_split(
        wine.drop(columns=[QUALITY, TARGET]),
        wine[TARGET],
        test_size=0.25,
        random_state=42,
        stratify=wine[TARGET]
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Обучение моделей
    models = {
        'kNN': KNeighborsClassifier(),
        'DT': DecisionTreeClassifier(random_state=42),
        'SVC': SVC(),
    }

    # 4. Оценка качества
    summary = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        print(f"Оценка модели {name}:\n", classification_report(y_test, y_pred))
        summary.append({
            'Модель': name,
            'Precision (good)': precision_score(y_test, y_pred),
            'Recall (good)': recall_score(y_test, y_pred),
            'F1 (good)': f1_score(y_test, y_pred),
            'F1 (macro)': f1_score(y_test, y_pred, average='macro'),
        })

    # 5. Сводное сравнение
    print("Сводная таблица:")
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    solution()