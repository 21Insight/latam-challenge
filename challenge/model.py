import pandas as pd
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from datetime import datetime
import joblib


class DelayModel:

    def __init__(
        self
    ):
        """
        Constructor for DelayModel class. Loads pre-trained model.
        """
        self._model = joblib.load('modelo_train.pkl')

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        # top 10 features
        top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

        # One-hot encode categorical features
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )

        # Add missing columns to features
        missing_columns = set(top_10_features) - set(features.columns)
        for col in missing_columns:
            features[col] = 0

        # Select top 10 features

        def get_period_day(date: str) -> str:
            """
            Get period of day from a given date.

            Args:
                date (str): date in string format.

            Returns:
                str: period of day (mañana, tarde, noche).
            """
            date_time = pd.to_datetime(date).time()
            if date_time.between_time('05:00', '11:59'):
                return 'mañana'
            elif date_time.between_time('12:00', '18:59'):
                return 'tarde'
            else:
                return 'noche'

        def is_high_season(fecha: str) -> int:
            """
            Check if a given date is in high season.

            Args:
                fecha (str): date in string format.

            Returns:
                int: 1 if in high season, 0 otherwise.
            """
            fecha = pd.to_datetime(fecha)
            fecha_año = fecha.year
            range1_min = pd.to_datetime(f'{fecha_año}-12-15')
            range1_max = pd.to_datetime(f'{fecha_año}-12-31')
            range2_min = pd.to_datetime(f'{fecha_año}-01-01')
            range2_max = pd.to_datetime(f'{fecha_año}-03-03')
            range3_min = pd.to_datetime(f'{fecha_año}-07-15')
            range3_max = pd.to_datetime(f'{fecha_año}-07-31')
            range4_min = pd.to_datetime(f'{fecha_año}-09-11')
            range4_max = pd.to_datetime(f'{fecha_año}-09-30')
            if ((fecha >= range1_min) & (fecha <= range1_max) |
                (fecha >= range2_min) & (fecha <= range2_max) |
                (fecha >= range3_min) & (fecha <= range3_max) |
                    (fecha >= range4_min) & (fecha <= range4_max)):
                return 1
            else:
                return 0

        def get_min_diff(data: pd.Series) -> float:
            """
            Get the difference in minutes between two dates.

            Args:
                data (pd.Series): row of data containing two dates.

            Returns:
                float: difference in minutes.
            """
            fecha_o = pd.to_datetime(data['Fecha-O'])
            fecha_i = pd.to_datetime(data['Fecha-I'])
            min_diff = (fecha_o - fecha_i).total_seconds() / 60
            return min_diff

        class Model:
            def __init__(self):
                """
                Constructor for Model class. Initializes XGBoost classifier.
                """
                self._model = xgb.XGBClassifier(
                    random_state=1, learning_rate=0.01)

            def preprocess(self, data: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
                """
                Prepare raw data for training or prediction.

                Args:
                    data (pd.DataFrame): raw data.
                    target_column (str, optional): if set, the target is returned.

                Returns:
                    pd.DataFrame: features and target.
                """
                top_10_features = [
                    "OPERA_Latin American Wings",
                    "MES_7",
                    "MES_10",
                    "OPERA_Grupo LATAM",
                    "MES_12",
                    "TIPOVUELO_I",
                    "MES_4",
                    "MES_11",
                    "OPERA_Sky Airline",
                    "OPERA_Copa Air"
                ]
                features = data[top_10_features]

                if target_column is not None:
                    data['period_day'] = data['Fecha-I'].apply(get_period_day)
                    data['high_season'] = data['Fecha-I'].apply(is_high_season)
                    data['min_diff'] = data.apply(get_min_diff, axis=1)

                    threshold_in_minutes = 15
                    data['delay'] = np.where(
                        data['min_diff'] > threshold_in_minutes, 1, 0)
                    target = data[target_column].astype(int)

                    self.fit(features, target)

                    return pd.concat([features, target], axis=1)
                else:
                    return features

            def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
                """
                Train the XGBoost classifier.

                Args:
                    features (pd.DataFrame): features.
                    target (pd.Series): target.
                """
                n_y0 = len(target[target == 0])
                n_y1 = len(target[target == 1])
                scale = n_y0 / n_y1

                x_train, x_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.33, random_state=42)

                x_train2, x_test2, y_train2, y_test2 = train_test_split(
                    features[top_10_features], target, test_size=0.33, random_state=42)

                xgb_model_2 = xgb.XGBClassifier(
                    random_state=1, learning_rate=0.01, scale_pos_weight=scale)
                xgb_model_2.fit(x_train2, y_train2)
                joblib.dump(xgb_model_2, 'modelo_train.pkl')
                self._model = xgb_model_2

            def predict(self, features: pd.DataFrame) -> List[int]:
                predictions = self._model.predict(features)
                return predictions.tolist()
