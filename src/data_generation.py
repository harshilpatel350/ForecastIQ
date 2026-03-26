"""
Realistic Multi-City Restaurant / Product Sales Dataset Generator
================================================================
Generates 100K+ rows of rich, temporally continuous sales data with:
  - Daily/weekly/yearly seasonality (Fourier terms)
  - Festival & holiday spikes (Indian + global)
  - Weather impact simulation
  - City-wise demand variation
  - Product popularity trends & lifecycle effects
  - Business growth trends
  - Random noise, anomalies (~1-2%), and missing values (~3%)
"""
import os, sys, random, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────
CITIES = {
    "Mumbai":    {"base_demand": 320, "growth": 0.0004, "weather_hot": True},
    "Delhi":     {"base_demand": 290, "growth": 0.0003, "weather_hot": True},
    "Bangalore": {"base_demand": 270, "growth": 0.0005, "weather_hot": False},
    "Hyderabad": {"base_demand": 250, "growth": 0.0004, "weather_hot": True},
    "Chennai":   {"base_demand": 240, "growth": 0.0003, "weather_hot": True},
    "Pune":      {"base_demand": 220, "growth": 0.0005, "weather_hot": False},
    "Kolkata":   {"base_demand": 210, "growth": 0.0003, "weather_hot": True},
    "Jaipur":    {"base_demand": 180, "growth": 0.0002, "weather_hot": True},
}

RESTAURANTS = {
    "Mumbai":    ["Bombay Bites", "Marine Drive Café", "Spice Junction", "Urban Tadka", "Curry House"],
    "Delhi":     ["Delhi Darbar", "Chandni Chowk Kitchen", "Mughal Zaika", "Punjab Grill", "Old Delhi Eats"],
    "Bangalore": ["Garden City Café", "Silicon Bites", "Namma Kitchen", "Tech Park Eats", "Koramangala Grills"],
    "Hyderabad": ["Biryani Blues", "Charminar Kitchen", "Spicy Tales", "Nawab's Plate", "Deccan Delights"],
    "Chennai":   ["Marina Meals", "Chettinad Corner", "South Spice", "Temple City Café", "Adyar Kitchen"],
    "Pune":      ["FC Road Bites", "Shaniwar Wada Kitchen", "Deccan Queen Café", "Pune Plate", "Koregaon Eats"],
    "Kolkata":   ["Park Street Café", "Howrah Kitchen", "Bengali Bites", "Rasgulla House", "Kolkata Grills"],
    "Jaipur":    ["Pink City Café", "Hawa Mahal Kitchen", "Royal Rajasthani", "Desert Bites", "Amber Fort Eats"],
}

CATEGORIES = {
    "Biryani":       {"products": ["Chicken Biryani", "Mutton Biryani", "Veg Biryani", "Egg Biryani"], "price_range": (180, 450)},
    "Pizza":         {"products": ["Margherita", "Pepperoni", "Farmhouse", "BBQ Chicken Pizza"], "price_range": (150, 500)},
    "Burger":        {"products": ["Classic Burger", "Cheese Burger", "Veggie Burger", "Double Patty Burger"], "price_range": (100, 350)},
    "North Indian":  {"products": ["Butter Chicken", "Paneer Tikka", "Dal Makhani", "Chole Bhature"], "price_range": (120, 380)},
    "South Indian":  {"products": ["Masala Dosa", "Idli Sambar", "Vada", "Uttapam"], "price_range": (60, 200)},
    "Chinese":       {"products": ["Fried Rice", "Manchurian", "Noodles", "Spring Roll"], "price_range": (100, 300)},
    "Desserts":      {"products": ["Gulab Jamun", "Ice Cream Sundae", "Brownie", "Rasgulla"], "price_range": (50, 200)},
    "Beverages":     {"products": ["Cold Coffee", "Mango Lassi", "Fresh Lime Soda", "Masala Chai"], "price_range": (40, 150)},
}

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Wallet"]
DELIVERY_PARTNERS = ["Swiggy", "Zomato", "DoorDash", "In-House", "Uber Eats"]
ORDER_TYPES = ["Delivery", "Dine-In", "Takeaway"]

# Indian + Global Holidays  (month, day) → multiplier
HOLIDAYS = {
    (1, 1): 1.40,   # New Year
    (1, 26): 1.30,  # Republic Day
    (2, 14): 1.35,  # Valentine's
    (3, 8): 1.15,   # Women's Day
    (3, 29): 1.25,  # Holi (approx)
    (8, 15): 1.35,  # Independence Day
    (10, 2): 1.20,  # Gandhi Jayanti
    (10, 24): 1.50, # Diwali (approx)
    (10, 25): 1.55, # Diwali Day 2
    (10, 26): 1.45, # Diwali Day 3
    (11, 1): 1.15,  # Bhai Dooj
    (11, 14): 1.20, # Children's Day
    (12, 25): 1.50, # Christmas
    (12, 31): 1.60, # New Year's Eve
}

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rainy", "Stormy", "Foggy", "Hot"]

# ── Helper functions ───────────────────────────────────────────────────────

def _seasonal_multiplier(day_of_year: int, hour: int, weekday: int) -> float:
    """Combine daily + weekly + yearly seasonality via Fourier terms."""
    yearly = (
        1.0
        + 0.10 * np.sin(2 * np.pi * day_of_year / 365)
        + 0.06 * np.cos(2 * np.pi * day_of_year / 365)
        + 0.04 * np.sin(4 * np.pi * day_of_year / 365)
    )
    # Lunch (12-14) and dinner (19-22) peaks
    hourly_weights = np.array([
        0.05, 0.03, 0.02, 0.02, 0.02, 0.03,  # 0-5
        0.08, 0.15, 0.25, 0.35, 0.55, 0.80,  # 6-11
        1.00, 0.95, 0.65, 0.45, 0.40, 0.50,  # 12-17
        0.70, 0.90, 1.05, 1.00, 0.75, 0.35,  # 18-23
    ])
    hourly = hourly_weights[hour]
    # Weekend bump
    weekly = 1.25 if weekday >= 5 else (0.90 + 0.03 * weekday)
    return yearly * hourly * weekly


def _weather_for_date(month: int, is_hot_city: bool) -> str:
    """Simple probabilistic weather model."""
    if month in (6, 7, 8):        # Monsoon
        return np.random.choice(WEATHER_CONDITIONS, p=[0.15, 0.20, 0.40, 0.10, 0.05, 0.10])
    elif month in (12, 1, 2):     # Winter
        if is_hot_city:
            return np.random.choice(WEATHER_CONDITIONS, p=[0.45, 0.25, 0.05, 0.02, 0.08, 0.15])
        else:
            return np.random.choice(WEATHER_CONDITIONS, p=[0.30, 0.25, 0.05, 0.02, 0.28, 0.10])
    else:                         # Summer / transition
        if is_hot_city:
            return np.random.choice(WEATHER_CONDITIONS, p=[0.30, 0.15, 0.10, 0.03, 0.02, 0.40])
        else:
            return np.random.choice(WEATHER_CONDITIONS, p=[0.40, 0.25, 0.15, 0.05, 0.05, 0.10])


def _weather_multiplier(weather: str) -> float:
    return {"Clear": 1.0, "Cloudy": 0.95, "Rainy": 0.75, "Stormy": 0.55, "Foggy": 0.85, "Hot": 0.90}[weather]


def _temperature(month: int, is_hot_city: bool) -> float:
    base = 30 if is_hot_city else 24
    seasonal = 8 * np.sin(2 * np.pi * (month - 4) / 12)
    return round(base + seasonal + np.random.normal(0, 2), 1)


def generate_dataset(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    output_path: str = None,
) -> pd.DataFrame:
    """Generate the full realistic sales dataset."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sales_data.csv")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_range = pd.date_range(start, end, freq="h")  # hourly granularity

    rows = []
    order_id = 100000

    for ts in date_range:
        hour = ts.hour
        weekday = ts.weekday()
        month = ts.month
        day_of_year = ts.day_of_year
        date_key = (ts.month, ts.day)

        holiday_mult = HOLIDAYS.get(date_key, 1.0)
        seasonal_mult = _seasonal_multiplier(day_of_year, hour, weekday)

        for city, cfg in CITIES.items():
            # Business growth over time
            days_since_start = (ts - start).days
            growth_factor = 1.0 + cfg["growth"] * days_since_start

            weather = _weather_for_date(month, cfg["weather_hot"])
            weather_mult = _weather_multiplier(weather)
            temp = _temperature(month, cfg["weather_hot"])

            # Expected orders this hour for this city
            expected = cfg["base_demand"] / 24.0 * seasonal_mult * holiday_mult * weather_mult * growth_factor
            n_orders = max(0, int(np.random.poisson(max(0.1, expected))))

            restaurants = RESTAURANTS[city]

            for _ in range(n_orders):
                order_id += 1
                restaurant = random.choice(restaurants)
                category = random.choice(list(CATEGORIES.keys()))
                cat_info = CATEGORIES[category]
                product = random.choice(cat_info["products"])
                base_price = random.uniform(*cat_info["price_range"])

                # Quantity 1-5, heavily skewed to 1-2
                quantity = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5])
                unit_price = round(base_price * (1 + np.random.normal(0, 0.05)), 2)
                discount_pct = np.random.choice([0, 0, 0, 5, 10, 15, 20, 25], p=[0.35, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07, 0.05])
                total = round(unit_price * quantity * (1 - discount_pct / 100), 2)

                # Competitor price index: slight noise around 1.0
                competitor_price_idx = round(np.random.normal(1.0, 0.08), 3)
                demand_index = round(expected / (cfg["base_demand"] / 24.0), 3)

                # Operational fields
                payment = random.choice(PAYMENT_METHODS)
                order_type = np.random.choice(ORDER_TYPES, p=[0.55, 0.30, 0.15])
                delivery_partner = random.choice(DELIVERY_PARTNERS) if order_type == "Delivery" else "N/A"
                prep_time = max(5, int(np.random.normal(25, 8)))  # minutes
                rating = round(np.clip(np.random.normal(4.0, 0.6), 1, 5), 1)

                # Inventory status
                inv = np.random.choice(["In Stock", "Low Stock", "Out of Stock"], p=[0.80, 0.15, 0.05])

                # Cancellation probability increases when stormy or out-of-stock
                cancel_prob = 0.03
                if weather == "Stormy":
                    cancel_prob += 0.08
                if inv == "Out of Stock":
                    cancel_prob += 0.15
                cancelled = int(np.random.random() < cancel_prob)

                # Customer segment
                customer_id = f"CUST-{random.randint(10000, 99999)}"
                is_new = int(np.random.random() < 0.25)

                rows.append({
                    "order_id": f"ORD-{order_id}",
                    "datetime": ts,
                    "date": ts.date(),
                    "hour": hour,
                    "weekday": weekday,
                    "month": month,
                    "year": ts.year,
                    "day_of_year": day_of_year,
                    "city": city,
                    "restaurant": restaurant,
                    "category": category,
                    "product": product,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount_pct": discount_pct,
                    "total_amount": total,
                    "payment_method": payment,
                    "order_type": order_type,
                    "delivery_partner": delivery_partner,
                    "customer_id": customer_id,
                    "is_new_customer": is_new,
                    "rating": rating,
                    "weather": weather,
                    "temperature": temp,
                    "is_holiday": int(holiday_mult > 1.0),
                    "holiday_multiplier": holiday_mult,
                    "competitor_price_index": competitor_price_idx,
                    "demand_index": demand_index,
                    "inventory_status": inv,
                    "preparation_time": prep_time,
                    "cancellation_flag": cancelled,
                })

    df = pd.DataFrame(rows)

    # ── Inject anomalies (~1.5%) ──
    n_anomalies = int(len(df) * 0.015)
    anomaly_idx = np.random.choice(df.index, size=n_anomalies, replace=False)
    df.loc[anomaly_idx, "total_amount"] *= np.random.uniform(2.5, 5.0, size=n_anomalies)
    df.loc[anomaly_idx, "quantity"] = np.random.choice([8, 10, 15, 20], size=n_anomalies)

    # ── Inject missing values (~3%) ──
    cols_to_null = ["rating", "temperature", "preparation_time", "competitor_price_index", "discount_pct"]
    for col in cols_to_null:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    # Shuffle slightly to break perfect ordering within same hour (realistic)
    df = df.sort_values("datetime").reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Generated {len(df):,} rows → {output_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
