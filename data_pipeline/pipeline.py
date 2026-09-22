import sqlite3
from pathlib import Path
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
HEADERS = {"User-Agent": "Mozilla/5.0"}

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def category_links():
    soup = get_soup(BASE_URL)
    links = []
    for a in soup.select(".side_categories ul li ul li a"):
        href = a.get("href")
        name = a.get_text(strip=True)
        if href and name:
            links.append((name, BASE_URL + href))
    return links


def parse_book(article, category):
    title = article.h3.a.get("title", article.h3.get_text(strip=True))
    price_text = article.select_one(".price_color").get_text(strip=True)
    rating_class = article.select_one(".star-rating").get("class", [])
    rating_text = next((x for x in rating_class if x in RATING_MAP), "")
    availability = article.select_one(".availability").get_text(" ", strip=True)

    price_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", price_text)
    price_gbp = float(price_match.group(1)) if price_match else None
    rating = RATING_MAP.get(rating_text)
    in_stock = "In stock" in availability

    return {
        "title": title,
        "price_gbp": price_gbp,
        "rating": rating,
        "availability": availability,
        "in_stock": in_stock,
        "category": category,
    }


def scrape_category(category, url, max_pages=50):
    rows = []
    current = url

    for _ in range(max_pages):
        soup = get_soup(current)
        for article in soup.select("article.product_pod"):
            rows.append(parse_book(article, category))

        next_link = soup.select_one("li.next a")
        if not next_link:
            break
        current = current.rsplit("/", 1)[0] + "/" + next_link["href"]

    return rows


def clean_data(rows):
    df = pd.DataFrame(rows)

    numeric_cols = ["price_gbp", "rating"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    df = df.dropna(subset=["title", "category"]).reset_index(drop=True)
    return df


def create_database(df, db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("PRAGMA foreign_keys = ON")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY(category_id) REFERENCES categories(category_id)
        )
    """)

    cur.execute("DELETE FROM books")
    cur.execute("DELETE FROM categories")

    for category in sorted(df["category"].unique()):
        cur.execute(
            "INSERT INTO categories(category_name) VALUES (?)",
            (category,)
        )

    category_map = dict(
        cur.execute("SELECT category_name, category_id FROM categories").fetchall()
    )

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            category_map[row["category"]],
        ))

    con.commit()
    return con


def run_queries(con, output_path):
    queries = {
        "Q1 SELECT WHERE": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4
            LIMIT 10;
        """,
        "Q2 ORDER BY": """
            SELECT title, price_inr
            FROM books
            ORDER BY price_inr DESC
            LIMIT 10;
        """,
        "Q3 DISTINCT": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating;
        """,
        "Q4 BETWEEN": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            ORDER BY price_gbp;
        """,
        "Q5 IN": """
            SELECT title, rating
            FROM books
            WHERE rating IN (4, 5)
            ORDER BY rating DESC;
        """,
        "Q6 JOIN": """
            SELECT c.category_name, b.title, b.rating, b.price_gbp
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            ORDER BY b.rating DESC, c.category_name, b.title
            LIMIT 10;
        """
    }

    with open(output_path, "w", encoding="utf-8") as f:
        for name, query in queries.items():
            f.write(f"\n{name}\n{'='*70}\n{query.strip()}\n")
            result = pd.read_sql_query(query, con)
            f.write(result.to_string(index=False))
            f.write("\n")


def pandas_equivalence(con):
    books = pd.read_sql_query("SELECT * FROM books", con)
    categories = pd.read_sql_query("SELECT * FROM categories", con)

    sql_join = pd.read_sql_query("""
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, c.category_name, b.title
        LIMIT 10;
    """, con)

    merged = books.merge(categories, on="category_id", how="inner")
    merged = merged[["category_name", "title", "rating", "price_gbp"]]
    merged = merged.sort_values(
        ["rating", "category_name", "title"],
        ascending=[False, True, True]
    ).head(10).reset_index(drop=True)

    sql_join = sql_join.reset_index(drop=True)

    print("\nJOIN comparison using pd.read_sql():")
    print(sql_join.to_string(index=False))
    print("\nEquivalent JOIN using pd.merge():")
    print(merged.to_string(index=False))
    print("\nEquivalent:", sql_join.equals(merged))


def main():
    root = Path(__file__).resolve().parent
    db_path = root / "books.db"
    output_path = root / "query_outputs.txt"

    print("Discovering categories...")
    categories = category_links()

    # Use at least three categories and continue until >=60 rows.
    selected = categories[:3]
    print("Selected categories:", [x[0] for x in selected])

    rows = []
    for name, url in selected:
        print("Scraping:", name)
        rows.extend(scrape_category(name, url))

    # If the first three categories unexpectedly provide fewer than 60,
    # add categories until the acceptance threshold is met.
    idx = 3
    while len(rows) < 60 and idx < len(categories):
        name, url = categories[idx]
        print("Adding category:", name)
        rows.extend(scrape_category(name, url))
        idx += 1

    df = clean_data(rows)
    if len(df) < 60 or df["category"].nunique() < 3:
        raise RuntimeError("Could not collect at least 60 books across 3 categories.")

    print(f"\nCollected {len(df)} rows across {df['category'].nunique()} categories.")
    print(df.head())
    print("\nFixed conversion rate:", GBP_TO_INR)

    con = create_database(df, db_path)
    run_queries(con, output_path)
    pandas_equivalence(con)
    con.close()

    print("\nCreated:", db_path)
    print("Created:", output_path)


if __name__ == "__main__":
    main()
