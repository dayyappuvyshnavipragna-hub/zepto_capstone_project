import sqlite3
from pathlib import Path
import re

import requests
import pandas as pd
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
HEADERS = {"User-Agent": "Mozilla/5.0"}

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# ============================================================
# WEBSITE REQUEST
# ============================================================

def get_soup(url):
    """Download a webpage and return a BeautifulSoup object."""

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    return BeautifulSoup(
        response.text,
        "html.parser"
    )


# ============================================================
# CATEGORY DISCOVERY
# ============================================================

def category_links():
    """Get all book category names and URLs."""

    soup = get_soup(BASE_URL)

    links = []

    for a in soup.select(
        ".side_categories ul li ul li a"
    ):
        href = a.get("href")
        name = a.get_text(strip=True)

        if href and name:
            links.append(
                (
                    name,
                    BASE_URL + href
                )
            )

    return links


# ============================================================
# BOOK PARSING
# ============================================================

def parse_book(article, category):
    """
    Extract one book from a product article.

    Raw fields retained:
    - title
    - price_gbp
    - star_rating
    - availability
    - category

    Cleaned fields:
    - rating
    - in_stock
    """

    # -------------------------
    # Title
    # -------------------------

    title = article.h3.a.get(
        "title",
        article.h3.get_text(strip=True)
    )

    # -------------------------
    # Price
    # -------------------------

    price_text = article.select_one(
        ".price_color"
    ).get_text(strip=True)

    price_match = re.search(
        r"([0-9]+(?:\.[0-9]+)?)",
        price_text
    )

    if price_match:
        price_gbp = float(
            price_match.group(1)
        )
    else:
        price_gbp = None

    # -------------------------
    # Star rating
    # -------------------------

    rating_element = article.select_one(
        ".star-rating"
    )

    rating_class = rating_element.get(
        "class",
        []
    )

    star_rating = next(
        (
            value
            for value in rating_class
            if value in RATING_MAP
        ),
        ""
    )

    rating = RATING_MAP.get(
        star_rating
    )

    # -------------------------
    # Availability
    # -------------------------

    availability = article.select_one(
        ".availability"
    ).get_text(
        " ",
        strip=True
    )

    in_stock = "In stock" in availability

    # -------------------------
    # Return record
    # -------------------------

    return {
        "title": title,
        "price_gbp": price_gbp,
        "star_rating": star_rating,
        "rating": rating,
        "availability": availability,
        "in_stock": in_stock,
        "category": category,
    }


# ============================================================
# CATEGORY SCRAPER
# ============================================================

def scrape_category(category, url, max_pages=50):
    """
    Scrape all books from one category.

    Continues through pagination until:
    - there is no next page, or
    - max_pages is reached.
    """

    rows = []

    current = url

    for _ in range(max_pages):

        soup = get_soup(current)

        articles = soup.select(
            "article.product_pod"
        )

        for article in articles:

            rows.append(
                parse_book(
                    article,
                    category
                )
            )

        next_link = soup.select_one(
            "li.next a"
        )

        if not next_link:
            break

        current = (
            current.rsplit("/", 1)[0]
            + "/"
            + next_link["href"]
        )

    return rows


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(rows):
    """
    Clean scraped data.

    Cleaning decisions:
    - price_gbp is converted to numeric.
    - rating is converted from One-Five to 1-5.
    - numeric parsing failures are median-imputed.
    - in_stock is converted to boolean.
    - price_inr uses fixed 105.50 conversion.
    - rows missing title/category are removed.
    """

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No books were scraped."
        )

    print("\nCleaning data...")

    # --------------------------------------------------------
    # Price conversion
    # --------------------------------------------------------

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    price_missing = df[
        "price_gbp"
    ].isna().sum()

    # --------------------------------------------------------
    # Rating conversion
    # --------------------------------------------------------

    df["rating"] = df[
        "star_rating"
    ].map(RATING_MAP)

    rating_missing = df[
        "rating"
    ].isna().sum()

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    for col in [
        "price_gbp",
        "rating"
    ]:

        if df[col].isna().any():

            median_value = df[
                col
            ].median()

            if pd.isna(median_value):

                raise RuntimeError(
                    f"Cannot median-impute column: {col}"
                )

            df[col] = df[
                col
            ].fillna(median_value)

    # --------------------------------------------------------
    # Availability → Boolean
    # --------------------------------------------------------

    df["in_stock"] = df[
        "in_stock"
    ].astype(bool)

    # --------------------------------------------------------
    # Fixed GBP → INR conversion
    # --------------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"]
        * GBP_TO_INR
    ).round(2)

    # --------------------------------------------------------
    # Remove rows missing essential text fields
    # --------------------------------------------------------

    before_drop = len(df)

    df = df.dropna(
        subset=[
            "title",
            "category"
        ]
    ).reset_index(drop=True)

    dropped_rows = (
        before_drop - len(df)
    )

    # --------------------------------------------------------
    # Cleaning report
    # --------------------------------------------------------

    print(
        "Price parsing failures:",
        price_missing
    )

    print(
        "Rating parsing failures:",
        rating_missing
    )

    print(
        "Rows removed because title/category "
        "was missing:",
        dropped_rows
    )

    print(
        "Numeric parsing failures are handled "
        "using median imputation."
    )

    print(
        "Fixed currency conversion:",
        "1 GBP = 105.50 INR"
    )

    print(
        "Final cleaned rows:",
        len(df)
    )

    return df


# ============================================================
# DATABASE CREATION
# ============================================================

def create_database(df, db_path):
    """
    Create a normalized SQLite database.

    Tables:
    - categories
    - books

    Relationship:
    books.category_id →
    categories.category_id
    """

    con = sqlite3.connect(
        db_path
    )

    cur = con.cursor()

    # Enable foreign-key enforcement.
    cur.execute(
        "PRAGMA foreign_keys = ON"
    )

    # --------------------------------------------------------
    # Recreate tables for a clean run
    # --------------------------------------------------------

    cur.execute(
        "DROP TABLE IF EXISTS books"
    )

    cur.execute(
        "DROP TABLE IF EXISTS categories"
    )

    # --------------------------------------------------------
    # Categories table
    # --------------------------------------------------------

    cur.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    # --------------------------------------------------------
    # Books table
    # --------------------------------------------------------

    cur.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            star_rating TEXT NOT NULL,
            rating INTEGER NOT NULL,
            availability TEXT NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY(category_id)
                REFERENCES categories(category_id)
        )
    """)

    # --------------------------------------------------------
    # Insert categories
    # --------------------------------------------------------

    for category in sorted(
        df["category"].unique()
    ):

        cur.execute(
            """
            INSERT INTO categories(category_name)
            VALUES (?)
            """,
            (category,)
        )

    # --------------------------------------------------------
    # Build category lookup
    # --------------------------------------------------------

    category_map = dict(
        cur.execute(
            """
            SELECT
                category_name,
                category_id
            FROM categories
            """
        ).fetchall()
    )

    # --------------------------------------------------------
    # Insert books
    # --------------------------------------------------------

    for _, row in df.iterrows():

        cur.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                star_rating,
                rating,
                availability,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                row["star_rating"],
                int(row["rating"]),
                row["availability"],
                int(row["in_stock"]),
                category_map[
                    row["category"]
                ],
            )
        )

    con.commit()

    return con


# ============================================================
# SQL QUERIES
# ============================================================

def run_queries(con, output_path):
    """
    Run SQL queries covering:

    - SELECT
    - WHERE
    - ORDER BY
    - LIMIT
    - DISTINCT
    - BETWEEN
    - IN
    - JOIN

    Results are saved to query_outputs.txt.
    """

    queries = {

        # ----------------------------------------------------
        # Q1 SELECT + WHERE + LIMIT
        # ----------------------------------------------------

        "Q1 SELECT WHERE LIMIT": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            WHERE rating >= 4
            LIMIT 10;
        """,

        # ----------------------------------------------------
        # Q2 ORDER BY + LIMIT
        # ----------------------------------------------------

        "Q2 ORDER BY LIMIT": """
            SELECT
                title,
                price_inr
            FROM books
            ORDER BY price_inr DESC
            LIMIT 10;
        """,

        # ----------------------------------------------------
        # Q3 DISTINCT
        # ----------------------------------------------------

        "Q3 DISTINCT": """
            SELECT DISTINCT
                rating
            FROM books
            ORDER BY rating;
        """,

        # ----------------------------------------------------
        # Q4 BETWEEN
        # ----------------------------------------------------

        "Q4 BETWEEN": """
            SELECT
                title,
                price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            ORDER BY price_gbp;
        """,

        # ----------------------------------------------------
        # Q5 IN
        # ----------------------------------------------------

        "Q5 IN": """
            SELECT
                title,
                rating
            FROM books
            WHERE rating IN (4, 5)
            ORDER BY rating DESC;
        """,

        # ----------------------------------------------------
        # Q6 JOIN
        # ----------------------------------------------------

        "Q6 JOIN": """
            SELECT
                c.category_name,
                b.title,
                b.star_rating,
                b.rating,
                b.price_gbp,
                b.price_inr
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY
                b.rating DESC,
                c.category_name,
                b.title
            LIMIT 10;
        """,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        for name, query in queries.items():

            f.write(
                f"\n{name}\n"
            )

            f.write(
                "=" * 70
            )

            f.write("\n")

            f.write(
                query.strip()
            )

            f.write("\n\n")

            # Use pd.read_sql exactly as required.
            result = pd.read_sql(
                query,
                con
            )

            f.write(
                result.to_string(
                    index=False
                )
            )

            f.write("\n")

    print(
        "\nSQL query results saved to:",
        output_path
    )


# ============================================================
# SQL JOIN VS PANDAS MERGE
# ============================================================

def pandas_equivalence(con):
    """
    Reproduce the SQL JOIN using pd.merge()
    and verify that both results match.
    """

    # --------------------------------------------------------
    # Read database tables using pd.read_sql()
    # --------------------------------------------------------

    books = pd.read_sql(
        "SELECT * FROM books",
        con
    )

    categories = pd.read_sql(
        "SELECT * FROM categories",
        con
    )

    # --------------------------------------------------------
    # SQL JOIN
    # --------------------------------------------------------

    sql_join = pd.read_sql(
        """
        SELECT
            c.category_name,
            b.title,
            b.rating,
            b.price_gbp
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY
            b.rating DESC,
            c.category_name,
            b.title
        LIMIT 10;
        """,
        con
    )

    # --------------------------------------------------------
    # Equivalent Pandas JOIN
    # --------------------------------------------------------

    merged = books.merge(
        categories,
        on="category_id",
        how="inner"
    )

    merged = merged[
        [
            "category_name",
            "title",
            "rating",
            "price_gbp"
        ]
    ]

    merged = merged.sort_values(
        [
            "rating",
            "category_name",
            "title"
        ],
        ascending=[
            False,
            True,
            True
        ]
    ).head(10).reset_index(
        drop=True
    )

    sql_join = sql_join.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "SQL JOIN using pd.read_sql()"
    )

    print(
        "========================================"
    )

    print(
        sql_join.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "Equivalent JOIN using pd.merge()"
    )

    print(
        "========================================"
    )

    print(
        merged.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    matches = sql_join.equals(
        merged
    )

    print(
        "\nSQL JOIN and pd.merge() match:",
        matches
    )

    if not matches:
        raise RuntimeError(
            "SQL JOIN and pd.merge() results do not match."
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    root = Path(
        __file__
    ).resolve().parent

    db_path = root / "books.db"

    output_path = (
        root / "query_outputs.txt"
    )

    # --------------------------------------------------------
    # Discover categories
    # --------------------------------------------------------

    print(
        "========================================"
    )

    print(
        "ZEPT0 DATA PIPELINE"
    )

    print(
        "========================================"
    )

    print(
        "\nDiscovering categories..."
    )

    categories = category_links()

    if len(categories) < 3:

        raise RuntimeError(
            "Fewer than 3 categories were found."
        )

    # --------------------------------------------------------
    # Start with 3 categories
    # --------------------------------------------------------

    selected = categories[:3]

    print(
        "Selected categories:",
        [x[0] for x in selected]
    )

    # --------------------------------------------------------
    # Scrape
    # --------------------------------------------------------

    rows = []

    for name, url in selected:

        print(
            "\nScraping:",
            name
        )

        category_rows = scrape_category(
            name,
            url
        )

        print(
            "Books collected:",
            len(category_rows)
        )

        rows.extend(
            category_rows
        )

    # --------------------------------------------------------
    # Add categories until ≥60 books
    # --------------------------------------------------------

    idx = 3

    while (
        len(rows) < 60
        and idx < len(categories)
    ):

        name, url = categories[idx]

        print(
            "\nAdding category:",
            name
        )

        category_rows = scrape_category(
            name,
            url
        )

        print(
            "Books collected:",
            len(category_rows)
        )

        rows.extend(
            category_rows
        )

        idx += 1

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_data(
        rows
    )

    # --------------------------------------------------------
    # Acceptance criteria
    # --------------------------------------------------------

    if len(df) < 60:

        raise RuntimeError(
            f"Only {len(df)} books were collected. "
            "At least 60 are required."
        )

    if df["category"].nunique() < 3:

        raise RuntimeError(
            "At least 3 categories are required."
        )

    required_columns = [
        "title",
        "price_gbp",
        "star_rating",
        "rating",
        "availability",
        "in_stock",
        "category",
        "price_inr",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Missing required columns: "
            + str(missing_columns)
        )

    # --------------------------------------------------------
    # Show summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "SCRAPING SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Total books:",
        len(df)
    )

    print(
        "Total categories:",
        df["category"].nunique()
    )

    print(
        "Categories:",
        sorted(
            df["category"].unique()
        )
    )

    print(
        "Fixed conversion rate:",
        f"1 GBP = {GBP_TO_INR} INR"
    )

    print(
        "\nFirst 5 cleaned rows:"
    )

    print(
        df.head().to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Create database
    # --------------------------------------------------------

    con = create_database(
        df,
        db_path
    )

    print(
        "\nSQLite database created:"
    )

    print(
        db_path
    )

    # --------------------------------------------------------
    # Run SQL queries
    # --------------------------------------------------------

    run_queries(
        con,
        output_path
    )

    # --------------------------------------------------------
    # Verify SQL JOIN with pd.merge()
    # --------------------------------------------------------

    pandas_equivalence(
        con
    )

    # --------------------------------------------------------
    # Close database
    # --------------------------------------------------------

    con.close()

    print(
        "\n========================================"
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "========================================"
    )

    print(
        "Database:",
        db_path
    )

    print(
        "Query output:",
        output_path
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()