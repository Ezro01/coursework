import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path='C:/Все папки по жизни/Универ/6 семестр/Курсовая ТиМП/Датасет_для_курсовой_продакшен.csv'):
    """Загрузка и предобработка данных"""
    df = pd.read_csv(file_path, delimiter=';', decimal='.', parse_dates=["Дата"])
    df, next_date = add_empty_rows_for_next_date(df)
    df = new_columns_for_learning_models(df)
    df = prepare_final_features(df)
    return df, next_date

def add_empty_rows_for_next_date(df, date_col='Дата', product_col='Товар'):
    last_date = df[date_col].max()
    next_date = last_date + timedelta(days=1)
    products = df[product_col].unique()

    new_rows = pd.DataFrame(columns=df.columns)

    for product in products:
        new_row = {
            date_col: next_date,
            product_col: product
        }
        # Все остальные колонки будут NaN
        new_rows = pd.concat([
            new_rows,
            pd.DataFrame([new_row])
        ], ignore_index=True)

    df_with_next_date = pd.concat([df, new_rows], ignore_index=True)
    df_with_next_date = df_with_next_date.sort_values(by=['Дата', 'Товар'])

    # Объединяем с исходными данными
    return df_with_next_date, next_date

def new_columns_for_learning_models(df):

    ## Сортируем данные по дате (очень важно!)
    df = df.sort_values(by=['Товар', 'Дата'])

    # Лаги (значения за предыдущие периоды)
    df['Продано_1д_назад'] = df.groupby(['Товар'])['Продано'].shift(1)
    df['Поступило_1д_назад'] = df.groupby(['Товар'])['Поступило'].shift(1)
    df['Остаток_1д_назад'] = df.groupby(['Товар'])['Остаток'].shift(1)
    df['КоличествоЧеков_1д_назад'] = df.groupby(['Товар'])['Остаток'].shift(1)
    df['Заказ_1д_назад'] = df.groupby(['Товар'])['Остаток'].shift(1)

    df['Продано_частота'] = df.groupby(['Товар'])['Продано'].transform(
        lambda x: (x > 0).astype(int).shift(1))
    df['Продано_частота_7д'] = df.groupby(['Товар'])['Продано_частота'].transform(
        lambda x: x.rolling(window=7, min_periods=7).sum())
    df['Продано_частота_14д'] = df.groupby(['Товар'])['Продано_частота'].transform(
        lambda x: x.rolling(window=14, min_periods=14).sum())
    df['Продано_частота_21д'] = df.groupby(['Товар'])['Продано_частота'].transform(
        lambda x: x.rolling(window=21, min_periods=21).sum())

    df['Продано_темп_7д'] = df.groupby(['Товар'])['Продано'].transform(
        lambda x: x.rolling(window=7, min_periods=7).mean().shift(1))
    df['Продано_темп_14д'] = df.groupby(['Товар'])['Продано'].transform(
        lambda x: x.rolling(window=14, min_periods=14).mean().shift(1))
    df['Продано_темп_21д'] = df.groupby(['Товар'])['Продано'].transform(
        lambda x: x.rolling(window=21, min_periods=21).mean().shift(1))

    # Лаги (значения за предыдущие периоды)
    df['ПроданоСеть_1д_назад'] = df.groupby(['Товар'])['ПроданоСеть'].shift(1)
    df['ПоступилоСеть_1д_назад'] = df.groupby(['Товар'])['ПоступилоСеть'].shift(1)
    df['ОстатокСеть_1д_назад'] = df.groupby(['Товар'])['ОстатокСеть'].shift(1)
    df['КоличествоЧековСеть_1д_назад'] = df.groupby(['Товар'])['Остаток'].shift(1)

    df['ПроданоСеть_частота'] = df.groupby(['Товар'])['ПроданоСеть'].transform(
        lambda x: (x > 0).astype(int).shift(1))
    df['ПроданоСеть_частота_7д'] = df.groupby(['Товар'])['ПроданоСеть_частота'].transform(
        lambda x: x.rolling(window=7, min_periods=7).sum())
    df['ПроданоСеть_частота_14д'] = df.groupby(['Товар'])['ПроданоСеть_частота'].transform(
        lambda x: x.rolling(window=14, min_periods=14).sum())
    df['ПроданоСеть_частота_21д'] = df.groupby(['Товар'])['ПроданоСеть_частота'].transform(
        lambda x: x.rolling(window=21, min_periods=21).sum())

    df['ПроданоСеть_темп_7д'] = df.groupby(['Товар'])['ПроданоСеть'].transform(
        lambda x: x.rolling(window=7, min_periods=7).mean().shift(1))
    df['ПроданоСеть_темп_14д'] = df.groupby(['Товар'])['ПроданоСеть'].transform(
        lambda x: x.rolling(window=14, min_periods=14).mean().shift(1))
    df['ПроданоСеть_темп_21д'] = df.groupby(['Товар'])['ПроданоСеть'].transform(
        lambda x: x.rolling(window=21, min_periods=21).mean().shift(1))

    # df = df[~(df['is_future'] == 1)]
    df = df.drop(['Продано_частота', 'ПроданоСеть_частота'], axis=1)

    return df.sort_values(by=['Дата', 'Товар'])

def prepare_final_features(df):
    """Подготовка финального датафрейма для предсказания"""
    df = df.drop([
        'ПроданоСеть', 'ПоступилоСеть', 'ОстатокСеть',
        'КоличествоЧеков', 'КоличествоЧековСеть',
        'Продано', 'Заказ', 'Поступило', 'Остаток'
    ], axis=1)

    df['День_недели'] = df['Дата'].dt.dayofweek
    df['День'] = df['Дата'].dt.day
    df['Месяц'] = df['Дата'].dt.month
    df['Год'] = df['Дата'].dt.year
    df['Выходной'] = df['День_недели'].isin([5, 6]).astype(int)
    df['Акция'] = df['Акция'].fillna(0)

    return df

def predict_sales(df_predict):
    """Запуск предсказания модели"""
    test_preduction = df_predict[['Дата', 'Товар']].copy()
    df = df_predict.drop('Дата', axis=1)

    # Обработка категориальных и числовых фичей (ваш код)
    object_cols = ['Поступило_1д_назад', 'ПоступилоСеть_1д_назад']
    for col in object_cols:
        if col in df.columns:
            # Convert to string first to handle any mixed types
            df[col] = df[col].astype(str)
            # Then try to convert to numeric where possible
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NA if needed
            df[col] = df[col].fillna(0)

    categorical_columns = ['Товар']
    label_encoder = LabelEncoder()
    df['Товар'] = label_encoder.fit_transform(df['Товар'])

    numerical_columns = [
        'Продано_1д_назад', 'Поступило_1д_назад', 'Остаток_1д_назад',
        "КоличествоЧеков_1д_назад", "Заказ_1д_назад",
        "Продано_частота_7д", "Продано_частота_14д", "Продано_частота_21д",
        "Продано_темп_7д", "Продано_темп_14д", "Продано_темп_21д",
        "ПроданоСеть_1д_назад", "ПоступилоСеть_1д_назад", "ОстатокСеть_1д_назад",
        "КоличествоЧековСеть_1д_назад", "ПроданоСеть_частота_7д",
        "ПроданоСеть_частота_14д", "ПроданоСеть_частота_21д",
        "ПроданоСеть_темп_7д", "ПроданоСеть_темп_14д", "ПроданоСеть_темп_21д"
    ]

    minmax_scaler = MinMaxScaler()
    df[numerical_columns] = minmax_scaler.fit_transform(df[numerical_columns])

    cat_features = ['Товар', 'Акция', 'Выходной', 'Год', 'Месяц', 'День', 'День_недели']
    df[cat_features] = df[cat_features].astype(str)

    # Загрузка модели и предсказание
    model = cb.CatBoostRegressor()
    model.load_model('C:/Все папки по жизни/Универ/6 семестр/Курсовая ТиМП/Веса_модели.cbm')

    pool = cb.Pool(data=df, cat_features=cat_features)
    y_pred = model.predict(pool)

    test_preduction['Прогноз'] = np.ceil(y_pred).astype(int)
    return test_preduction

if __name__ == "__main__":
    # Пример использования (для теста)
    df, next_date = load_and_preprocess_data()
    df_predict = df[df['Дата'] == next_date]
    result = predict_sales(df_predict)
    print(result)