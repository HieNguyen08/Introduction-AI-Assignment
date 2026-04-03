"""
Smoke tests for data_pipeline.py — validates core cleaning and feature functions.

Run:  python -m pytest tests/test_data_pipeline.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure modules/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.data_pipeline import (
    clean_vietnam_weather,
    clean_hotel_reviews,
    clean_travel_ratings,
    clean_hotel_bookings,
    clean_world_cities,
    haversine,
    build_distance_matrix,
    build_cost_matrix,
    build_travel_time_matrix,
    build_places_dataframe,
    build_weather_probability_table,
    _validate_dataframe,
)


# ── Fixtures: minimal DataFrames that mirror real dataset schemas ──────

@pytest.fixture
def weather_df():
    return pd.DataFrame({
        "province": ["Ha Noi", "Ha Noi", "Da Nang", "Da Nang"],
        "date": ["2020-01-01", "2020-07-15", "2020-01-01", "2020-07-15"],
        "max": [18.0, 36.0, 22.0, 38.5],
        "min": [12.0, 28.0, 16.0, 27.0],
        "wind": [3.0, 2.5, 4.0, 3.5],
        "rain": [0.0, 15.0, 0.5, 0.0],
        "humidity": [75.0, 85.0, 70.0, 92.0],
        "cloud": [60.0, 80.0, 50.0, 30.0],
        "pressure": [1015.0, 1005.0, 1012.0, 1008.0],
    })


@pytest.fixture
def reviews_df():
    return pd.DataFrame({
        "Positive_Review": ["Great hotel", "Nice view", "No Positive"],
        "Negative_Review": ["No Negative", "Room was small", "Terrible service"],
        "Reviewer_Score": [9.0, 6.5, 3.0],
    })


@pytest.fixture
def ratings_df():
    data = {f"Category {i}": [float(i % 5), float((i + 1) % 5)] for i in range(1, 25)}
    return pd.DataFrame(data)


@pytest.fixture
def bookings_df():
    return pd.DataFrame({
        "hotel": ["Resort Hotel", "City Hotel"],
        "is_canceled": [0, 1],
        "lead_time": [30, 60],
        "arrival_date_year": [2020, 2020],
        "arrival_date_month": ["July", "January"],
        "arrival_date_day_of_month": [15, 5],
        "stays_in_weekend_nights": [2, 0],
        "stays_in_week_nights": [3, 2],
        "adults": [2, 1],
        "children": [1, 0],
        "babies": [0, 0],
        "adr": [120.0, 80.0],
        "agent": ["123", "NULL"],
        "company": ["NULL", "NULL"],
    })


@pytest.fixture
def cities_df():
    return pd.DataFrame({
        "City": ["Paris", "Tokyo"],
        "culture_rating": ["8.5", "9.0"],
        "adventure_score": ["6.0", "7.5"],
        "Latitude": ["48.8566", "35.6762"],
        "Longitude": ["2.3522", "139.6503"],
    })


# ── Validation tests ──────────────────────────────────────────────────

class TestValidation:
    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="expected pandas DataFrame"):
            _validate_dataframe([1, 2, 3], "test")

    def test_rejects_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_dataframe(pd.DataFrame(), "test")

    def test_warns_on_missing_columns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="data_pipeline"):
            _validate_dataframe(
                pd.DataFrame({"a": [1]}), "test", required_cols=["a", "b"]
            )
        assert "missing expected columns" in caplog.text


# ── Clean function tests ──────────────────────────────────────────────

class TestCleanVietnamWeather:
    def test_output_shape_and_columns(self, weather_df):
        result = clean_vietnam_weather(weather_df)
        assert len(result) == 4
        for col in ["province", "temp_max", "rain_mm", "is_rainy", "outdoor_suitable",
                     "year", "month", "season"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_weather_labels_are_binary(self, weather_df):
        result = clean_vietnam_weather(weather_df)
        for col in ["is_rainy", "is_humid", "is_hot", "outdoor_suitable"]:
            assert set(result[col].unique()).issubset({0, 1})

    def test_no_null_in_numeric_cols(self, weather_df):
        result = clean_vietnam_weather(weather_df)
        for col in ["temp_max", "temp_min", "rain_mm", "humidity"]:
            if col in result.columns:
                assert result[col].isna().sum() == 0


class TestCleanHotelReviews:
    def test_output_has_sentiment(self, reviews_df):
        result = clean_hotel_reviews(reviews_df)
        assert "sentiment" in result.columns
        assert "sentiment_binary" in result.columns
        assert "full_review" in result.columns

    def test_sentiment_values(self, reviews_df):
        result = clean_hotel_reviews(reviews_df)
        valid = {"negative", "neutral", "positive", "very_positive"}
        assert set(result["sentiment"].dropna().unique()).issubset(valid)

    def test_text_features(self, reviews_df):
        result = clean_hotel_reviews(reviews_df)
        assert "review_word_count" in result.columns
        assert (result["review_word_count"] >= 0).all()


class TestCleanTravelRatings:
    def test_category_rename(self, ratings_df):
        result = clean_travel_ratings(ratings_df)
        assert "Churches" in result.columns
        assert "Category 1" not in result.columns

    def test_aggregated_features(self, ratings_df):
        result = clean_travel_ratings(ratings_df)
        assert "top_category" in result.columns
        assert "avg_rating" in result.columns


class TestCleanHotelBookings:
    def test_derived_columns(self, bookings_df):
        result = clean_hotel_bookings(bookings_df)
        assert "total_nights" in result.columns
        assert "total_cost" in result.columns
        assert "budget_level" in result.columns
        assert "season" in result.columns

    def test_filters_zero_guests(self, bookings_df):
        bad = bookings_df.copy()
        bad.loc[0, "adults"] = 0
        bad.loc[0, "children"] = 0
        bad.loc[0, "babies"] = 0
        result = clean_hotel_bookings(bad)
        assert len(result) < len(bad)


class TestCleanWorldCities:
    def test_numeric_conversion(self, cities_df):
        result = clean_world_cities(cities_df)
        assert result["Latitude"].dtype in [np.float64, np.float32]


# ── Feature engineering tests ─────────────────────────────────────────

class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine(10.0, 106.0, 10.0, 106.0) == 0.0

    def test_known_distance(self):
        # Ha Noi to Ho Chi Minh ~ 1,140 km (approximate)
        d = haversine(21.0285, 105.8542, 10.8231, 106.6297)
        assert 1100 < d < 1200


class TestMatrices:
    def test_distance_matrix_shape(self):
        names, matrix = build_distance_matrix()
        n = len(names)
        assert matrix.shape == (n, n)
        from modules.data_pipeline import VN_TOURIST_PLACES
        assert n == len(VN_TOURIST_PLACES)
        # Symmetric
        np.testing.assert_array_almost_equal(matrix, matrix.T)
        # Zero diagonal
        np.testing.assert_array_equal(np.diag(matrix), np.zeros(n))

    def test_cost_matrix_proportional(self):
        names, cost = build_cost_matrix()
        _, dist = build_distance_matrix()
        from modules.data_pipeline import TRAVEL_COST_PER_KM
        np.testing.assert_array_almost_equal(cost, dist * TRAVEL_COST_PER_KM)

    def test_time_matrix_proportional(self):
        names, time_m = build_travel_time_matrix()
        _, dist = build_distance_matrix()
        from modules.data_pipeline import TRAVEL_AVG_SPEED_KMH
        np.testing.assert_array_almost_equal(time_m, dist / TRAVEL_AVG_SPEED_KMH)


class TestPlacesDataframe:
    def test_places_count_and_columns(self):
        from modules.data_pipeline import VN_TOURIST_PLACES
        df = build_places_dataframe(use_osm=False)
        assert len(df) == len(VN_TOURIST_PLACES)
        for col in ["place_name", "latitude", "longitude", "category",
                     "province", "entry_fee_vnd", "visit_duration_hours",
                     "opening_hour", "closing_hour"]:
            assert col in df.columns

    def test_opening_before_closing(self):
        df = build_places_dataframe()
        assert (df["opening_hour"] < df["closing_hour"]).all()


class TestWeatherProbability:
    def test_probability_table(self, weather_df):
        cleaned = clean_vietnam_weather(weather_df)
        probs = build_weather_probability_table(cleaned)
        assert probs is not None
        assert "p_rain" in probs.columns
        # Probabilities between 0 and 1
        assert (probs["p_rain"] >= 0).all() and (probs["p_rain"] <= 1).all()
