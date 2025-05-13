import json

from bs4 import BeautifulSoup
from selenium import webdriver

# Set up Selenium and load the page
driver = webdriver.Chrome()
driver.get("http://books.toscrape.com/catalogue/category/books/travel_2/index.html")
driver.implicitly_wait(5)

# Parse the page with BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
books_data = []

for book in soup.select("article.product_pod"):
    # Title
    title = (
        book.select_one("h3 a")["title"].strip() if book.select_one("h3 a") else None
    )
    # Price
    price = (
        book.select_one("div.product_price p.price_color").get_text(strip=True)
        if book.select_one("div.product_price p.price_color")
        else None
    )
    # Stock
    stock = (
        book.select_one("p.instock.availability").get_text(strip=True)
        if book.select_one("p.instock.availability")
        else None
    )
    # Star rating
    rating_p = book.find("p", class_="star-rating")
    star_rating = None
    if rating_p:
        classes = rating_p.get("class", [])
        rating_word = next((c for c in classes if c in RATING_MAP), None)
        star_rating = RATING_MAP[rating_word] if rating_word else None

    books_data.append(
        {
            "title": title,
            "price": price,
            "star_rating": star_rating,
            "stock": stock,
        }
    )

# Print the extracted data in JSON format
"""
ensure_accii=False: Ensure Currency Symbol in Price 
"""
print(json.dumps(books_data, indent=2, ensure_ascii=False))
