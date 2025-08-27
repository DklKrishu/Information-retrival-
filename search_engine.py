import json
import os
import random
import time
import pickle
import re
from collections import deque
from urllib.parse import urljoin
from urllib import robotparser
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup, Tag
import schedule
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import webbrowser
import threading

import logging
logging.getLogger('selenium').setLevel(logging.WARNING)  # Only show warnings or higher
os.environ['WDM_LOG_LEVEL'] = '0'  # Suppress webdriver-manager logs

COVENTRY_PUREPORTAL_URL = "https://pureportal.coventry.ac.uk/en/organisations/fbl-school-of-economics-finance-and-accounting/publications"
BASE_URL = "https://pureportal.coventry.ac.uk"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
MAX_RETRIES = 3             # Number of times to retry failed page loads
PAGE_TIMEOUT = 30           # Timeout for page load in seconds
CRAWLED_DATA_FILE = "coventry_publications.json"
INDEX_FILE = "tfidf_index.pkl"
POLITE_DELAY = 2            # Minimum delay between requests to avoid overloading server

# Download stopwords data from nltk
nltk.download('stopwords')

# Helper function: Extract authors from publication page
# ---------------------------
def fetch_publication_author(soup, base_url):
    authors_data = []

    # Find the paragraph element that contains author information
    persons_p = soup.select_one('p.relations.persons')
    if not persons_p:
        return []  # No authors found

    # Iterate through the contents of the paragraph
    for element in persons_p.contents:
        if isinstance(element, Tag) and element.name == 'a':
            # If the element is a link, extract name and profile URL
            name = element.get_text(strip=True)
            url = urljoin(base_url, str(element.get('href', '')))
            if name:
                authors_data.append({'name': name, 'url': url})
        elif isinstance(element, str):
            # If plain text, it may contain unlinked author names (e.g., collaborators)
            potential_names = element.split(',')
            for name_part in potential_names:
                clean_name = name_part.strip(' ,')
                if clean_name:
                    authors_data.append({'name': clean_name, 'url': None})
    return authors_data

# Helper function: Extract abstract from publication page
# ---------------------------
def fetch_pulication_abstract(soup):
    # Locate the div containing the abstract
    abstract_div = soup.find('div', class_='rendering_researchoutput_abstractportal')
    if abstract_div:
        # The actual abstract text is inside a nested 'textblock' div
        text_block = abstract_div.find('div', class_='textblock')
        if text_block:
            return text_block.get_text(strip=True)
    return ''  # Return empty string if abstract not found

# Helper functions for indexing
# ---------------------------
def preprocess(text):
    """Lowercase, remove non-alphanumerics, remove stopwords."""
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return ' '.join(tokens)

def build_index():
    """Build TF-IDF index from crawled data and save to file."""
    print("\n--- Building TF-IDF Index ---")
    
    if not os.path.exists(CRAWLED_DATA_FILE):
        print(f"Error: {CRAWLED_DATA_FILE} not found. Run crawler first.")
        return
    
    with open(CRAWLED_DATA_FILE, "r", encoding="utf-8") as f:
        publications = json.load(f)
    
    corpus = []
    pub_urls = []
    titles = []
    pub_data = []

    for pub in publications:
        # Combine title + abstract + author names for better search
        authors_text = " ".join([a['name'] for a in pub.get("authors", [])])
        text = f"{pub.get('title','')} {pub.get('abstract','')} {authors_text}"
        corpus.append(preprocess(text))
        pub_urls.append(pub.get("url", "#"))
        titles.append(pub.get("title", ""))
        pub_data.append(pub)

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Save index to file
    index_data = {
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'pub_urls': pub_urls,
        'titles': titles,
        'publications': pub_data
    }
    
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"TF-IDF index built and saved to {INDEX_FILE}")
    print(f"Indexed {len(publications)} publications")

def load_index():
    """Load the saved TF-IDF index."""
    if not os.path.exists(INDEX_FILE):
        print(f"Index file {INDEX_FILE} not found. Building index...")
        build_index()
    
    with open(INDEX_FILE, 'rb') as f:
        index_data = pickle.load(f)
    
    return index_data

# Main crawler function
# ---------------------------
def crawl():
    print("Starting crawler with Selenium...")

    # ---------------------------
    # Setup Selenium Chrome driver options
    # ---------------------------
    options = Options()
    options.add_argument("--headless=new")  # Run in headless mode (no browser UI)
    options.add_argument(f"user-agent={USER_AGENT}")  # Set custom user agent
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")  # Suppress console logging
    options.add_argument("--silent")
    # Suppress automation messages
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option('useAutomationExtension', False)

    # Configure driver service to suppress log output
    service = Service(log_output=os.devnull)
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, PAGE_TIMEOUT)  # Explicit wait helper

    # ---------------------------
    # Parse robots.txt for crawling rules
    # ---------------------------
    rp = robotparser.RobotFileParser()
    robots_url = urljoin(BASE_URL, 'robots.txt')
    print(f"Fetching robots.txt from: {robots_url}")

    try:
        # Use Selenium to fetch robots.txt (can also use requests)
        driver.get(robots_url)
        robots_page_source = driver.page_source
        soup_robots = BeautifulSoup(robots_page_source, 'html.parser')
        robots_text_value = soup_robots.get_text()

        print("\n--- robots.txt content ---")
        print(robots_text_value)
        print("----------------------------------------------\n")

        # Parse robots.txt lines
        rp.parse(robots_text_value.splitlines())
        print("robots.txt parsed successfully.")
    except WebDriverException as e:
        print(f"Warning: Could not fetch or parse robots.txt. Proceeding with default settings. Error: {e}")

    # Determine crawl delay from robots.txt or default
    crawl_delay_duration = rp.crawl_delay(USER_AGENT)
    robots_delay_duration = int(crawl_delay_duration) if crawl_delay_duration else None
    user_minimum_delay = POLITE_DELAY

    if robots_delay_duration and robots_delay_duration > user_minimum_delay:
        min_effective_delay = robots_delay_duration
        print(f"Using Crawl-Delay from robots.txt: {min_effective_delay} seconds.")
    else:
        min_effective_delay = user_minimum_delay
        print(f"Using default delay: {min_effective_delay} seconds.")

    # Ensure the output directory exists
    data_dir = os.path.dirname(CRAWLED_DATA_FILE)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    try:
        # ---------------------------
        # PHASE 1: Discover all publication URLs
        # ---------------------------
        print("\n--- Phase 1: Discovering all publication URLs ---")
        publications_to_scrape = []
        queue = deque([COVENTRY_PUREPORTAL_URL])  # Start with the main publications page
        visited_urls = {COVENTRY_PUREPORTAL_URL}

        print("Scanning Pages...")

        while queue:
            current_url = queue.popleft()

            # Skip disallowed URLs according to robots.txt
            if not rp.can_fetch(USER_AGENT, current_url):
                print(f"\nSkipping disallowed URL (from robots.txt): {current_url}")
                continue

            success = False
            print(f"Visiting: {current_url}")

            # Retry mechanism for page loading
            for attempt in range(MAX_RETRIES):
                try:
                    driver.get(current_url)

                    # Handle cookie consent pop-up
                    if len(visited_urls) == 1:
                        try:
                            WebDriverWait(driver, 5).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))
                            ).click()
                        except Exception:
                            pass

                    # Wait for publication list to load
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.list-result-item")))
                    success = True
                    break
                except (TimeoutException, WebDriverException) as e:
                    print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed for {current_url}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print("Retrying...")
                        time.sleep(random.uniform(min_effective_delay, min_effective_delay + 2))

            if not success:
                print(f"All retries failed for {current_url}. Skipping page.")
                continue

            # Parse current page
            soup = BeautifulSoup(driver.page_source, "html.parser")
            pub_list = soup.find_all("li", class_="list-result-item")
            print(f"Found {len(pub_list)} publications on this page.")

            for pub_item in pub_list:
                if isinstance(pub_item, Tag):
                    title_tag = pub_item.find("h3", class_="title")
                    if isinstance(title_tag, Tag) and title_tag.a:
                        title = title_tag.get_text(strip=True)
                        pub_url = urljoin(BASE_URL, str(title_tag.a["href"]))
                        date_tag = pub_item.find("span", class_="date")
                        date = date_tag.get_text(strip=True) if date_tag else "N/A"
                        publications_to_scrape.append({"title": title, "url": pub_url, "date": date})
                        print(f"Found publication: {title} ({date}) - {pub_url}")

            # Check if there is a next page
            next_page_tag = soup.find("a", class_="nextLink")
            print(f"Next page link found: {next_page_tag is not None}")
            if isinstance(next_page_tag, Tag) and "href" in next_page_tag.attrs:
                next_page_url = urljoin(BASE_URL, str(next_page_tag["href"]))
                if next_page_url not in visited_urls:
                    visited_urls.add(next_page_url)
                    queue.append(next_page_url)

            # Polite crawling delay
            time.sleep(random.uniform(min_effective_delay, min_effective_delay + 2))

        print(f"--- Discovery complete. Found {len(publications_to_scrape)} publications to scrape. ---")

        # ---------------------------
        # PHASE 2: Scrape author details and abstract
        # ---------------------------
        print("\n--- Phase 2: Scraping author details and abstract for each publication ---")
        final_publications = []

        for pub_data in publications_to_scrape:
            if not rp.can_fetch(USER_AGENT, pub_data['url']):
                print(f"\nSkipping disallowed URL (from robots.txt): {pub_data['url']}")
                continue

            success = False
            print(f"\nProcessing publication: {pub_data['url']}")
            for attempt in range(MAX_RETRIES):
                try:
                    driver.get(pub_data["url"])
                    # Wait until authors paragraph is visible
                    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "p.relations.persons")))
                    detail_soup = BeautifulSoup(driver.page_source, 'html.parser')

                    # Extract authors and abstract
                    pub_data["authors"] = fetch_publication_author(detail_soup, BASE_URL)
                    pub_data["abstract"] = fetch_pulication_abstract(detail_soup)
                    success = True
                    print(f"Successfully scraped publication: {pub_data['title']}")
                    break
                except (TimeoutException, WebDriverException) as e:
                    print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed for {pub_data['url']}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print("Retrying...")
                        time.sleep(random.uniform(min_effective_delay, min_effective_delay + 2))

            if not success:
                # If all retries fail, store empty author/abstract
                print(f"All retries failed for {pub_data['url']}. Saving without author details and abstract.")
                pub_data["authors"] = []
                pub_data["abstract"] = ""

            final_publications.append(pub_data)
            # Polite delay between publication detail requests
            time.sleep(random.uniform(min_effective_delay, min_effective_delay + 2))

    finally:
        # Close the Selenium driver regardless of success or failure
        print("\nClosing Selenium driver.")
        driver.quit()
 # Save data to JSON file
    # ---------------------------
    with open(CRAWLED_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(final_publications, f, indent=4, ensure_ascii=False)

    print(f"\nCrawling complete. Found {len(final_publications)} publications.")
    print(f"Data saved to {CRAWLED_DATA_FILE}")

def crawl_and_index():
    """Combined function to crawl data and build index."""
    print(f"\n=== Starting scheduled crawl and index at {datetime.now()} ===")
    crawl()
    build_index()
    print(f"=== Crawl and index completed at {datetime.now()} ===")

def search(query, top_k=10):
    """Search function using saved TF-IDF index."""
    index_data = load_index()
    vectorizer = index_data['vectorizer']
    tfidf_matrix = index_data['tfidf_matrix']
    titles = index_data['titles']
    pub_urls = index_data['pub_urls']
    
    query_vec = vectorizer.transform([preprocess(query)])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        if scores[idx] > 0:  # Only return results with positive scores
            results.append({
                "title": titles[idx],
                "url": pub_urls[idx],
                "score": scores[idx]
            })
    return results

class SearchEngineGUI:
    def __init__(self, root):
        self.root = root
        self.setup_gui()
        
    def setup_gui(self):
        # Configure main window
        self.root.title("Vertical Search Engine")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Define colors for light theme
        bg_color = "#ffffff"
        secondary_bg = "#f8f9fa"
        primary_color = "#1a365d"
        accent_color = "#2563eb"
        hover_color = "#1d4ed8"
        text_color = "#374151"
        border_color = "#e5e7eb"
        success_color = "#059669"
        
        self.root.configure(bg=bg_color)
        
        # Main frame with minimal padding
        main_frame = tk.Frame(self.root, bg=bg_color, padx=20, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header section
        header_frame = tk.Frame(main_frame, bg=bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Title
        title_label = tk.Label(
            header_frame, 
            text="Vertical Search Engine",
            font=("Segoe UI", 32, "bold"),
            fg=primary_color,
            bg=bg_color
        )
        title_label.pack(anchor=tk.W)
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Discover academic publications from Faculty of Economics, Finance & Accounting",
            font=("Segoe UI", 14),
            fg=text_color,
            bg=bg_color
        )
        subtitle_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Search section with minimal card design
        search_card = tk.Frame(main_frame, bg=secondary_bg, relief=tk.FLAT, bd=1)
        search_card.pack(fill=tk.X, pady=(0, 15), ipady=12, ipadx=15)
        
        # Search label
        search_label = tk.Label(
            search_card,
            text="Enter your search query:",
            font=("Segoe UI", 16, "bold"),
            fg=primary_color,
            bg=secondary_bg
        )
        search_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Search input frame
        search_input_frame = tk.Frame(search_card, bg=secondary_bg)
        search_input_frame.pack(fill=tk.X)
        
        # Search entry without wrapper frame
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            search_input_frame,
            textvariable=self.search_var,
            font=("Segoe UI", 16),
            bg="#ffffff",
            fg=text_color,
            relief=tk.SOLID,
            bd=1,
            highlightthickness=0,
            insertwidth=2,
            insertbackground=text_color,
            insertofftime=300,
            insertontime=600
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), pady=0, ipady=8)
        self.search_entry.bind('<Return>', self.perform_search)
        
        # Search button with black text
        self.search_button = tk.Button(
            search_input_frame,
            text="🔍 Search",
            command=self.perform_search,
            font=("Segoe UI", 16, "bold"),
            bg=accent_color,
            fg="black",
            relief=tk.FLAT,
            bd=0,
            padx=25,
            pady=8,
            cursor="hand2",
            activebackground=hover_color,
            activeforeground="black"
        )
        self.search_button.pack(side=tk.RIGHT)
        
        # Results section
        results_label = tk.Label(
            main_frame,
            text="Search Results",
            font=("Segoe UI", 20, "bold"),
            fg=primary_color,
            bg=bg_color
        )
        results_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Results frame with border
        results_container = tk.Frame(main_frame, bg=border_color, relief=tk.SOLID, bd=1)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        # Results text widget
        self.results_text = scrolledtext.ScrolledText(
            results_container,
            wrap=tk.WORD,
            font=("Segoe UI", 12),
            bg="#ffffff",
            fg=text_color,
            state=tk.DISABLED,
            cursor="arrow",
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=15,
            selectbackground="#e0e7ff"
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Configure text tags with minimal spacing
        self.results_text.tag_configure("title", 
            font=("Segoe UI", 18, "bold"), 
            foreground=primary_color,
            spacing1=2,
            spacing3=1
        )
        self.results_text.tag_configure("link", 
            font=("Segoe UI", 14), 
            underline=True, 
            foreground=accent_color
        )
        self.results_text.tag_configure("author_link", 
            font=("Segoe UI", 12), 
            underline=True, 
            foreground=accent_color
        )
        self.results_text.tag_configure("abstract", 
            font=("Segoe UI", 12), 
            foreground=text_color,
            lmargin1=15,
            lmargin2=15,
            spacing3=1
        )
        self.results_text.tag_configure("score", 
            font=("Segoe UI", 12, "bold"), 
            foreground="#dc2626",
            background="#fef2f2",
            relief=tk.RAISED,
            borderwidth=1
        )
        self.results_text.tag_configure("number", 
            font=("Segoe UI", 16, "bold"), 
            foreground=success_color
        )
        self.results_text.tag_configure("info", 
            font=("Segoe UI", 14), 
            foreground=text_color
        )
        self.results_text.tag_configure("header", 
            font=("Segoe UI", 16, "bold"), 
            foreground=primary_color,
            spacing3=5
        )
        self.results_text.tag_configure("separator", 
            foreground=border_color,
            spacing1=2,
            spacing3=2
        )
        
        # Status bar
        status_frame = tk.Frame(main_frame, bg=secondary_bg, relief=tk.FLAT)
        status_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to search. Enter keywords and press Search or Enter.")
        status_bar = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 12),
            fg=text_color,
            bg=secondary_bg,
            anchor=tk.W,
            padx=12,
            pady=5
        )
        status_bar.pack(fill=tk.X)
        
        # Bind events
        self.results_text.bind("<Button-1>", self.on_text_click)
        self.results_text.bind("<Motion>", self.on_mouse_motion)
        
        # Focus on search entry
        self.search_entry.focus()
        
        # Store URLs for clickable links
        self.url_ranges = {}

    def on_mouse_motion(self, event):
        # Get mouse position
        mouse_pos = self.results_text.index(f"@{event.x},{event.y}")
        
        # Check if mouse is over a URL
        is_over_link = False
        for (start, end), url in self.url_ranges.items():
            if self.results_text.compare(mouse_pos, ">=", start) and self.results_text.compare(mouse_pos, "<", end):
                is_over_link = True
                break
        
        # Change cursor based on whether mouse is over a link
        if is_over_link:
            self.results_text.configure(cursor="hand2")
        else:
            self.results_text.configure(cursor="arrow")
    
    def perform_search(self, event=None):
        query = self.search_var.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a search query.")
            return
        
        # Disable search button and show loading
        self.search_button.configure(state="disabled", text="Searching...")
        self.status_var.set("Searching publications...")
        self.root.update()
        
        # Run search in a separate thread to prevent GUI freezing
        thread = threading.Thread(target=self.search_thread, args=(query,))
        thread.daemon = True
        thread.start()
    
    def search_thread(self, query):
        try:
            results = search(query, top_k=1000)
            # Update GUI in main thread
            self.root.after(0, self.display_results, results, query)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def display_results(self, results, query):
        # Clear previous results
        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.url_ranges.clear()
        
        if not results:
            # No results styling
            self.results_text.insert(tk.END, "🔍 No Results Found\n\n", "header")
            self.results_text.insert(tk.END, f"We couldn't find any publications matching '{query}'.\n\n", "info")
            self.results_text.insert(tk.END, "💡 Suggestions:\n", "info")
            self.results_text.insert(tk.END, "• Try different or more general keywords\n", "abstract")
            self.results_text.insert(tk.END, "• Check your spelling\n", "abstract")
            self.results_text.insert(tk.END, "• Use fewer search terms\n", "abstract")
        else:
            # Results header
            header = f"📚 Found {len(results)} publications matching '{query}'\n"
            self.results_text.insert(tk.END, header, "header")
            self.results_text.insert(tk.END, "─" * 80 + "\n\n", "separator")
            
            # Load index to get full publication data
            try:
                index_data = load_index()
                publications = index_data['publications']
                
                # Create URL to publication mapping
                url_to_pub = {pub['url']: pub for pub in publications}
            except Exception:
                url_to_pub = {}
            
            for i, result in enumerate(results, 1):
                # Result number
                self.results_text.insert(tk.END, f"{i:2d}. ", "number")
                
                # Title
                title = result.get('title', 'Untitled')
                self.results_text.insert(tk.END, f"{title}\n", "title")
                
                # Get full publication data for abstract
                url = result.get('url', '#')
                pub_data = url_to_pub.get(url, {})
                abstract = pub_data.get('abstract', '')
                
                # Show abstract if available
                if abstract:
                    # Truncate abstract if too long
                    if len(abstract) > 300:
                        abstract = abstract[:300] + "..."
                    self.results_text.insert(tk.END, f"{abstract}\n", "abstract")
                else:
                    self.results_text.insert(tk.END, "Abstract not available.\n", "abstract")
                
                # Authors with clickable links
                authors = pub_data.get('authors', [])
                if authors:
                    self.results_text.insert(tk.END, "👥 Authors: ", "abstract")
                    for j, author in enumerate(authors[:3]):
                        if j > 0:
                            self.results_text.insert(tk.END, ", ", "abstract")
                        
                        author_name = author['name']
                        author_url = author.get('url')
                        
                        if author_url:
                            # Clickable author name
                            author_start = self.results_text.index(tk.INSERT)
                            self.results_text.insert(tk.END, author_name, "author_link")
                            author_end = self.results_text.index(tk.INSERT)
                            self.url_ranges[(author_start, author_end)] = author_url
                        else:
                            # Non-clickable author name
                            self.results_text.insert(tk.END, author_name, "abstract")
                    
                    if len(authors) > 3:
                        self.results_text.insert(tk.END, ", et al.", "abstract")
                    self.results_text.insert(tk.END, "\n", "abstract")
                
                # Publication date
                pub_date = pub_data.get('date', '')
                if pub_date and pub_date != 'N/A':
                    self.results_text.insert(tk.END, f"📅 Published: {pub_date}\n", "abstract")
                
                # Clickable URL
                url_start = self.results_text.index(tk.INSERT)
                url_text = "🔗 View Full Publication\n"
                self.results_text.insert(tk.END, url_text, "link")
                url_end = self.results_text.index(tk.INSERT)
                
                # Store URL range for click handling
                self.url_ranges[(url_start, url_end)] = url
                
                # Relevance score with highlighting
                score = result.get('score', 0)
                self.results_text.insert(tk.END, f"⭐ Relevance Score: {score:.3f}", "score")
                self.results_text.insert(tk.END, "\n", "abstract")
                
                # Minimal separator between results
                self.results_text.insert(tk.END, "\n" + "─" * 40 + "\n", "separator")
            
            # Footer with tips
            footer = "\n💡 Tip: Click on blue links to open them in your browser."
            self.results_text.insert(tk.END, footer, "info")
        
        self.results_text.configure(state=tk.DISABLED)
        
        # Reset search button
        self.search_button.configure(state="normal", text="🔍 Search")
        self.status_var.set(f"✅ Search completed. Found {len(results)} results.")

    def show_error(self, error_msg):
        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error occurred during search:\n{error_msg}")
        self.results_text.configure(state=tk.DISABLED)
        
        self.search_button.configure(state="normal", text="Search")
        self.status_var.set("Error occurred during search.")
        messagebox.showerror("Search Error", f"An error occurred:\n{error_msg}")
    
    def on_text_click(self, event):
        # Get click position
        click_pos = self.results_text.index(f"@{event.x},{event.y}")
        
        # Check if click is on a URL
        for (start, end), url in self.url_ranges.items():
            if self.results_text.compare(click_pos, ">=", start) and self.results_text.compare(click_pos, "<", end):
                try:
                    webbrowser.open(url)
                    self.status_var.set(f"Opened: {url}")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open URL:\n{e}")
                break

def run_gui():
    """Run the GUI application."""
    root = tk.Tk()
    SearchEngineGUI(root)
    root.mainloop()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        # Run GUI mode
        run_gui()
    else:
        # Run initial crawl and build index
        #crawl()
        #build_index()
        
        # Schedule weekly crawl and index
        schedule.every().sunday.at("12:00").do(crawl_and_index)
        print("Scheduled weekly crawl and index initialized. Waiting for next run...")
        
        # Interactive search interface
        print("\nWelcome to the Vertical Search Engine!")
        print("Enter keywords to search for publications (type 'exit' to quit).")
        print("Or run with --gui flag to use the graphical interface.")

        while True:
            query = input("\nSearch query: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            results = search(query, top_k=10)
            if results:
                print(f"\nTop {len(results)} results:")
                for i, res in enumerate(results, 1):
                    print(f"{i}. {res['title']} - {res['url']} (Score: {res['score']:.3f})")
            else:
                print("No results found.")
