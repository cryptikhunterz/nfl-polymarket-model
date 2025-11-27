#!/usr/bin/env python3
"""
PFF (Pro Football Focus) Web Scraper

Scrapes premium stats from PFF using your subscription credentials.
Saves data to CSV files for use in the NFL Polymarket Model.

Usage:
    python pff_scraper.py --email "your@email.com" --password "yourpassword"
    python pff_scraper.py --email "your@email.com" --password "yourpassword" --interactive

Requirements:
    pip install playwright pandas
    playwright install chromium
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    import pandas as pd
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install playwright pandas")
    print("  playwright install chromium")
    sys.exit(1)


class PFFScraper:
    """Scraper for Pro Football Focus premium stats"""

    # PFF URLs for different stat pages
    PAGES = {
        'team_pass_blocking': 'https://premium.pff.com/nfl/teams/blocking/2024/reg?position=T,G,C',
        'team_pass_rush': 'https://premium.pff.com/nfl/teams/pass-rush/2024/reg',
        'team_offense': 'https://premium.pff.com/nfl/teams/offense/2024/reg',
        'team_defense': 'https://premium.pff.com/nfl/teams/defense/2024/reg',
        'qb_grades': 'https://premium.pff.com/nfl/players/passing/2024/reg?position=QB',
        'receiving': 'https://premium.pff.com/nfl/players/receiving/2024/reg',
        'rushing': 'https://premium.pff.com/nfl/players/rushing/2024/reg',
    }

    def __init__(self, email: str, password: str, headless: bool = True):
        """Initialize scraper with PFF credentials"""
        self.email = email
        self.password = password
        self.headless = headless
        self.browser = None
        self.page = None
        self.playwright = None

        # Output directory
        self.output_dir = Path(__file__).parent.parent / "data" / "pff"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start browser"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        self.page = self.browser.new_page(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        print("Browser started")

    def stop(self):
        """Stop browser"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        print("Browser stopped")

    def screenshot(self, name: str):
        """Take screenshot for debugging"""
        path = self.output_dir / f"{name}.png"
        self.page.screenshot(path=str(path))
        print(f"Screenshot saved: {path}")

    def login(self) -> bool:
        """Log into PFF"""
        print("Navigating to PFF login...")

        try:
            # Go to login page
            self.page.goto('https://www.pff.com/login', wait_until='networkidle', timeout=30000)
            time.sleep(2)
            self.screenshot('step1_login_page')

            # Look for login form
            # PFF's login can be tricky - they might use different selectors
            email_selectors = [
                'input[name="email"]',
                'input[type="email"]',
                '#email',
                'input[placeholder*="email" i]',
                'input[placeholder*="Email" i]',
            ]

            password_selectors = [
                'input[name="password"]',
                'input[type="password"]',
                '#password',
            ]

            # Find and fill email
            email_input = None
            for selector in email_selectors:
                try:
                    email_input = self.page.wait_for_selector(selector, timeout=3000)
                    if email_input:
                        break
                except:
                    continue

            if not email_input:
                print("Could not find email input field")
                self.screenshot('error_no_email_field')
                return False

            email_input.fill(self.email)
            print(f"Filled email: {self.email}")

            # Find and fill password
            password_input = None
            for selector in password_selectors:
                try:
                    password_input = self.page.wait_for_selector(selector, timeout=3000)
                    if password_input:
                        break
                except:
                    continue

            if not password_input:
                print("Could not find password input field")
                self.screenshot('error_no_password_field')
                return False

            password_input.fill(self.password)
            print("Filled password")

            self.screenshot('step2_filled_form')

            # Find and click login button
            login_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Log In")',
                'button:has-text("Sign In")',
                'button:has-text("Login")',
            ]

            login_button = None
            for selector in login_selectors:
                try:
                    login_button = self.page.wait_for_selector(selector, timeout=2000)
                    if login_button:
                        break
                except:
                    continue

            if not login_button:
                print("Could not find login button")
                self.screenshot('error_no_login_button')
                return False

            login_button.click()
            print("Clicked login button, waiting for redirect...")

            # Wait for navigation
            time.sleep(5)
            self.screenshot('step3_after_login')

            # Check if login succeeded
            current_url = self.page.url
            print(f"Current URL: {current_url}")

            if 'login' in current_url.lower():
                print("Still on login page - login may have failed")
                # Check for error messages
                error_el = self.page.query_selector('.error, .alert-danger, [class*="error"]')
                if error_el:
                    print(f"Error message: {error_el.inner_text()}")
                return False

            print("Login successful!")
            return True

        except Exception as e:
            print(f"Login error: {e}")
            self.screenshot('error_login_exception')
            return False

    def extract_table(self, page_name: str = None) -> pd.DataFrame:
        """Extract data table from current page"""
        print(f"Extracting table from current page...")

        try:
            # Wait for table to load
            time.sleep(3)

            # Try different table selectors
            table_selectors = [
                'table.kyber-table',
                'table[class*="table"]',
                'table',
                '.data-table table',
                '[class*="Table"] table',
            ]

            table = None
            for selector in table_selectors:
                try:
                    table = self.page.wait_for_selector(selector, timeout=5000)
                    if table:
                        print(f"Found table with selector: {selector}")
                        break
                except:
                    continue

            if not table:
                print("Could not find data table")
                self.screenshot('error_no_table')
                return pd.DataFrame()

            # Extract table HTML
            table_html = table.evaluate('el => el.outerHTML')

            # Parse with pandas
            dfs = pd.read_html(table_html)
            if dfs:
                df = dfs[0]
                print(f"Extracted table with {len(df)} rows, {len(df.columns)} columns")
                return df

            return pd.DataFrame()

        except Exception as e:
            print(f"Table extraction error: {e}")
            self.screenshot('error_table_extraction')
            return pd.DataFrame()

    def scrape_page(self, page_name: str) -> pd.DataFrame:
        """Navigate to a page and extract its table"""
        if page_name not in self.PAGES:
            print(f"Unknown page: {page_name}")
            print(f"Available pages: {list(self.PAGES.keys())}")
            return pd.DataFrame()

        url = self.PAGES[page_name]
        print(f"\nNavigating to {page_name}: {url}")

        try:
            self.page.goto(url, wait_until='networkidle', timeout=30000)
            time.sleep(3)

            df = self.extract_table(page_name)

            if not df.empty:
                # Save to CSV
                output_path = self.output_dir / f"{page_name}.csv"
                df.to_csv(output_path, index=False)
                print(f"Saved to {output_path}")

            return df

        except Exception as e:
            print(f"Error scraping {page_name}: {e}")
            self.screenshot(f'error_{page_name}')
            return pd.DataFrame()

    def scrape_all(self):
        """Scrape all PFF stat pages"""
        print("\n" + "=" * 60)
        print("SCRAPING ALL PFF PAGES")
        print("=" * 60)

        results = {}
        for page_name in self.PAGES:
            df = self.scrape_page(page_name)
            results[page_name] = df
            time.sleep(2)  # Be polite

        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE")
        print("=" * 60)
        print(f"\nFiles saved to: {self.output_dir}")

        for name, df in results.items():
            status = f"{len(df)} rows" if not df.empty else "FAILED"
            print(f"  {name}: {status}")

        return results

    def interactive_mode(self):
        """Interactive mode for manual navigation and extraction"""
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("\nCommands:")
        print("  <filename>      - Extract current page table to <filename>.csv")
        print("  screenshot      - Take screenshot")
        print("  url             - Show current URL")
        print("  list            - Show available preset pages")
        print("  goto <name>     - Navigate to preset page")
        print("  quit            - Exit")
        print("\nBrowser is open. Log in manually if needed, then extract tables.")
        print("-" * 60)

        while True:
            try:
                cmd = input("\n> ").strip()

                if not cmd:
                    continue

                if cmd == 'quit' or cmd == 'exit':
                    break

                elif cmd == 'screenshot':
                    self.screenshot(f'manual_{datetime.now().strftime("%H%M%S")}')

                elif cmd == 'url':
                    print(f"Current URL: {self.page.url}")

                elif cmd == 'list':
                    print("Available pages:")
                    for name, url in self.PAGES.items():
                        print(f"  {name}: {url}")

                elif cmd.startswith('goto '):
                    page_name = cmd[5:].strip()
                    if page_name in self.PAGES:
                        self.page.goto(self.PAGES[page_name], wait_until='networkidle', timeout=30000)
                        print(f"Navigated to {page_name}")
                    else:
                        print(f"Unknown page: {page_name}")

                else:
                    # Assume it's a filename - extract table
                    filename = cmd if cmd.endswith('.csv') else f"{cmd}.csv"
                    df = self.extract_table()

                    if not df.empty:
                        output_path = self.output_dir / filename
                        df.to_csv(output_path, index=False)
                        print(f"Saved {len(df)} rows to {output_path}")
                    else:
                        print("No table found on current page")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\nExiting interactive mode")


def main():
    parser = argparse.ArgumentParser(description='PFF Web Scraper')
    parser.add_argument('-e', '--email', required=True, help='PFF account email')
    parser.add_argument('-p', '--password', help='PFF account password (will prompt if not provided)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode (visible browser)')
    parser.add_argument('--page', help='Scrape specific page only')
    parser.add_argument('--all', action='store_true', help='Scrape all pages')

    args = parser.parse_args()

    # Get password
    password = args.password
    if not password:
        import getpass
        password = getpass.getpass('PFF Password: ')

    # Create scraper
    scraper = PFFScraper(
        email=args.email,
        password=password,
        headless=not args.interactive
    )

    try:
        scraper.start()

        # Login
        if not scraper.login():
            print("\nLogin failed. Try interactive mode to log in manually:")
            print(f"  python {sys.argv[0]} -e '{args.email}' -i")
            return

        if args.interactive:
            scraper.interactive_mode()
        elif args.page:
            scraper.scrape_page(args.page)
        elif args.all:
            scraper.scrape_all()
        else:
            # Default: scrape all
            scraper.scrape_all()

    finally:
        scraper.stop()


if __name__ == '__main__':
    main()
