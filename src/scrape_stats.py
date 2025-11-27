#!/usr/bin/env python3
"""
NFL Stats Scraper - PFF + Next Gen Stats

Scrapes premium stats from multiple sources:
- PFF (Pro Football Focus) - requires subscription
- NFL Next Gen Stats - free, public data

Usage:
    # Scrape Next Gen Stats only (free, test first)
    python scrape_stats.py --nextgen

    # Scrape PFF only
    python scrape_stats.py --pff --email "your@email.com" --password "yourpass"

    # Scrape everything
    python scrape_stats.py --all --email "your@email.com" --password "yourpass"

Requirements:
    pip install playwright pandas
    playwright install chromium
"""

import argparse
import time
import sys
import json
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


class BaseStatsScraper:
    """Base class for stats scrapers"""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser = None
        self.page = None
        self.playwright = None
        self.output_dir = Path(__file__).parent.parent / "data" / "stats"
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
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
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
        print(f"Screenshot: {path}")

    def extract_table(self) -> pd.DataFrame:
        """Extract data table from current page"""
        try:
            time.sleep(2)

            # Try different table selectors
            selectors = ['table', '.data-table', '[class*="Table"]']

            for selector in selectors:
                try:
                    table = self.page.wait_for_selector(selector, timeout=5000)
                    if table:
                        html = table.evaluate('el => el.outerHTML')
                        dfs = pd.read_html(html)
                        if dfs:
                            return dfs[0]
                except:
                    continue

            return pd.DataFrame()
        except Exception as e:
            print(f"Table extraction error: {e}")
            return pd.DataFrame()

    def extract_pff_table(self) -> pd.DataFrame:
        """Extract PFF data table - handles React-rendered tables"""
        try:
            time.sleep(3)  # Give React time to render

            # Try to get table data via JavaScript evaluation
            # PFF uses a data grid that might not be a standard HTML table
            table_data = self.page.evaluate('''
                () => {
                    // Try standard table first
                    const table = document.querySelector('table');
                    if (table) {
                        const headers = [];
                        const headerCells = table.querySelectorAll('thead th, thead td');
                        headerCells.forEach(cell => headers.push(cell.innerText.trim()));

                        const rows = [];
                        const bodyRows = table.querySelectorAll('tbody tr');
                        bodyRows.forEach(row => {
                            const cells = row.querySelectorAll('td');
                            const rowData = [];
                            cells.forEach(cell => rowData.push(cell.innerText.trim()));
                            if (rowData.length > 0) rows.push(rowData);
                        });

                        if (rows.length > 0) {
                            return { headers, rows };
                        }
                    }

                    // Try data grid (common in React apps)
                    const gridRows = document.querySelectorAll('[role="row"]');
                    if (gridRows.length > 0) {
                        const headers = [];
                        const headerRow = document.querySelector('[role="row"]:first-child, [role="columnheader"]');
                        if (headerRow) {
                            const headerCells = headerRow.querySelectorAll('[role="columnheader"], [role="cell"]');
                            headerCells.forEach(cell => headers.push(cell.innerText.trim()));
                        }

                        const rows = [];
                        gridRows.forEach((row, idx) => {
                            if (idx === 0) return; // Skip header
                            const cells = row.querySelectorAll('[role="cell"], [role="gridcell"]');
                            const rowData = [];
                            cells.forEach(cell => rowData.push(cell.innerText.trim()));
                            if (rowData.length > 0) rows.push(rowData);
                        });

                        if (rows.length > 0) {
                            return { headers, rows };
                        }
                    }

                    // Try any div-based table structure
                    const dataRows = document.querySelectorAll('[class*="TableRow"], [class*="table-row"], [class*="Row"]');
                    if (dataRows.length > 1) {
                        const rows = [];
                        dataRows.forEach((row, idx) => {
                            const cells = row.querySelectorAll('[class*="Cell"], [class*="cell"], td, span');
                            const rowData = [];
                            cells.forEach(cell => {
                                const text = cell.innerText.trim();
                                if (text) rowData.push(text);
                            });
                            if (rowData.length > 0) rows.push(rowData);
                        });

                        if (rows.length > 1) {
                            return { headers: rows[0], rows: rows.slice(1) };
                        }
                    }

                    return null;
                }
            ''')

            if table_data and table_data.get('rows'):
                headers = table_data.get('headers', [])
                rows = table_data['rows']

                if headers and len(headers) >= len(rows[0]):
                    df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
                else:
                    df = pd.DataFrame(rows)
                    if headers:
                        df.columns = headers[:len(df.columns)]

                return df

            # Fallback: Get entire page HTML and try pd.read_html
            print("  Trying fallback HTML extraction...")
            html = self.page.content()
            dfs = pd.read_html(html)
            if dfs:
                # Return the largest table
                largest = max(dfs, key=len)
                return largest

            return pd.DataFrame()

        except Exception as e:
            print(f"PFF table extraction error: {e}")
            return pd.DataFrame()


class NextGenStatsScraper(BaseStatsScraper):
    """Scraper for NFL Next Gen Stats (free, public)"""

    def __init__(self, headless: bool = True, season: int = 2025):
        super().__init__(headless)
        self.season = season

        # Next Gen Stats URLs (dynamically built with season)
        self.PAGES = {
            # Receiving - Separation stats
            'receiving_season': f'https://nextgenstats.nfl.com/stats/receiving/{season}/REG/all',
            'receiving_weekly': f'https://nextgenstats.nfl.com/stats/receiving/{season}/REG/{{week}}',

            # Passing stats
            'passing_season': f'https://nextgenstats.nfl.com/stats/passing/{season}/REG/all',
            'passing_weekly': f'https://nextgenstats.nfl.com/stats/passing/{season}/REG/{{week}}',

            # Rushing stats
            'rushing_season': f'https://nextgenstats.nfl.com/stats/rushing/{season}/REG/all',
            'rushing_weekly': f'https://nextgenstats.nfl.com/stats/rushing/{season}/REG/{{week}}',
        }

    def scrape_separation_stats(self) -> dict:
        """Scrape receiver separation stats"""
        print("\n" + "=" * 60)
        print("SCRAPING NEXT GEN STATS - SEPARATION")
        print("=" * 60)

        results = {}

        # Season totals
        print("\nFetching season totals...")
        url = self.PAGES['receiving_season']
        self.page.goto(url, wait_until='networkidle', timeout=30000)
        time.sleep(3)

        df = self.extract_ngs_table()
        if not df.empty:
            output_path = self.output_dir / "ngs_separation_season.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} rows to {output_path}")
            results['season'] = df

        # Weekly data (for moving averages)
        print("\nFetching weekly data...")
        weekly_data = []

        for week in range(1, 18):  # Weeks 1-17
            try:
                url = self.PAGES['receiving_weekly'].format(week=week)
                self.page.goto(url, wait_until='networkidle', timeout=15000)
                time.sleep(2)

                df = self.extract_ngs_table()
                if not df.empty:
                    df['week'] = week
                    weekly_data.append(df)
                    print(f"  Week {week}: {len(df)} rows")
            except Exception as e:
                print(f"  Week {week}: Failed ({e})")
                continue

        if weekly_data:
            weekly_df = pd.concat(weekly_data, ignore_index=True)
            output_path = self.output_dir / "ngs_separation_weekly.csv"
            weekly_df.to_csv(output_path, index=False)
            print(f"Saved {len(weekly_df)} total weekly rows")
            results['weekly'] = weekly_df

            # Calculate team-level moving averages
            team_ma = self.calculate_team_separation_ma(weekly_df)
            if not team_ma.empty:
                output_path = self.output_dir / "team_separation_ma.csv"
                team_ma.to_csv(output_path, index=False)
                print(f"Saved team separation moving averages")
                results['team_ma'] = team_ma

        return results

    def extract_ngs_table(self) -> pd.DataFrame:
        """Extract table from Next Gen Stats page"""
        try:
            # NGS uses a specific table structure
            table = self.page.wait_for_selector('table', timeout=10000)
            if not table:
                return pd.DataFrame()

            # Get all rows
            rows = self.page.query_selector_all('table tbody tr')
            data = []

            for row in rows:
                cells = row.query_selector_all('td')
                row_data = [cell.inner_text().strip() for cell in cells]
                if row_data:
                    data.append(row_data)

            # Get headers
            headers = []
            header_cells = self.page.query_selector_all('table thead th')
            for cell in header_cells:
                headers.append(cell.inner_text().strip())

            if data and headers:
                df = pd.DataFrame(data, columns=headers[:len(data[0])])
                return df

            return pd.DataFrame()

        except Exception as e:
            print(f"NGS table extraction error: {e}")
            return pd.DataFrame()

    def calculate_team_separation_ma(self, weekly_df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
        """Calculate team-level separation moving averages"""
        try:
            # Identify separation column (might be named differently)
            sep_cols = [c for c in weekly_df.columns if 'sep' in c.lower()]
            if not sep_cols:
                print("No separation column found")
                return pd.DataFrame()

            sep_col = sep_cols[0]

            # Convert to numeric
            weekly_df[sep_col] = pd.to_numeric(weekly_df[sep_col], errors='coerce')

            # Identify team column
            team_cols = [c for c in weekly_df.columns if 'team' in c.lower()]
            team_col = team_cols[0] if team_cols else 'Team'

            # Group by team and calculate rolling average
            team_data = []
            for team in weekly_df[team_col].unique():
                team_weeks = weekly_df[weekly_df[team_col] == team].sort_values('week')

                if len(team_weeks) >= window:
                    ma = team_weeks[sep_col].rolling(window=window).mean().iloc[-1]
                else:
                    ma = team_weeks[sep_col].mean()

                team_data.append({
                    'team': team,
                    f'separation_{window}wk_ma': round(ma, 2) if pd.notna(ma) else None,
                    'games_in_window': min(len(team_weeks), window)
                })

            return pd.DataFrame(team_data)

        except Exception as e:
            print(f"Moving average calculation error: {e}")
            return pd.DataFrame()

    def scrape_all(self) -> dict:
        """Scrape all Next Gen Stats"""
        results = {}
        results['separation'] = self.scrape_separation_stats()
        return results


class PFFScraper(BaseStatsScraper):
    """Scraper for Pro Football Focus (requires subscription)"""

    def __init__(self, email: str, password: str, headless: bool = True, season: int = 2025):
        super().__init__(headless)
        self.email = email
        self.password = password
        self.season = season

        # PFF URLs (dynamically built with season)
        self.PAGES = {
            'team_blocking': f'https://premium.pff.com/nfl/teams/blocking/{season}/reg',
            'team_pass_rush': f'https://premium.pff.com/nfl/teams/pass-rush/{season}/reg',
            'team_offense': f'https://premium.pff.com/nfl/teams/offense/{season}/reg',
            'team_defense': f'https://premium.pff.com/nfl/teams/defense/{season}/reg',
            'qb_passing': f'https://premium.pff.com/nfl/players/passing/{season}/reg?position=QB',
        }

    def login(self) -> bool:
        """Log into PFF"""
        print("\nLogging into PFF...")

        try:
            # Use domcontentloaded instead of networkidle for faster load
            self.page.goto('https://www.pff.com/login', wait_until='domcontentloaded', timeout=60000)
            time.sleep(3)

            # Take screenshot for debugging
            self.screenshot('pff_login_page')

            # Find and fill email - try multiple selectors
            email_selectors = [
                'input[type="email"]',
                'input[name="email"]',
                'input[id*="email"]',
                'input[placeholder*="email" i]'
            ]
            email_input = None
            for selector in email_selectors:
                try:
                    email_input = self.page.wait_for_selector(selector, timeout=5000)
                    if email_input:
                        break
                except:
                    continue

            if email_input:
                email_input.fill(self.email)
                print(f"  Filled email: {self.email}")
            else:
                print("  Could not find email input")
                return False

            # Find and fill password
            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[id*="password"]'
            ]
            password_input = None
            for selector in password_selectors:
                try:
                    password_input = self.page.wait_for_selector(selector, timeout=5000)
                    if password_input:
                        break
                except:
                    continue

            if password_input:
                password_input.fill(self.password)
                print("  Filled password")
            else:
                print("  Could not find password input")
                return False

            # Click login button
            login_selectors = [
                'button[type="submit"]',
                'button:has-text("Log In")',
                'button:has-text("Sign In")',
                'input[type="submit"]'
            ]
            login_btn = None
            for selector in login_selectors:
                try:
                    login_btn = self.page.wait_for_selector(selector, timeout=3000)
                    if login_btn:
                        break
                except:
                    continue

            if login_btn:
                login_btn.click()
                print("  Clicked login button")
            else:
                # Try pressing Enter
                password_input.press('Enter')
                print("  Pressed Enter to submit")

            # Wait for navigation
            time.sleep(8)
            self.screenshot('pff_after_login')

            if 'login' not in self.page.url.lower():
                print("Login successful!")
                return True

            print(f"Login failed - still on: {self.page.url}")
            return False

        except Exception as e:
            print(f"Login error: {e}")
            self.screenshot('pff_login_error')
            return False

    def scrape_team_stats(self) -> pd.DataFrame:
        """Scrape team-level PFF stats"""
        print("\n" + "=" * 60)
        print("SCRAPING PFF TEAM STATS")
        print("=" * 60)

        all_data = []

        # Scrape blocking stats (PBLK)
        print("\nFetching blocking stats...")
        try:
            self.page.goto(self.PAGES['team_blocking'], wait_until='domcontentloaded', timeout=60000)
            time.sleep(5)
            self.screenshot('pff_blocking_page')
            blocking_df = self.extract_pff_table()  # Use PFF-specific extraction
            print(f"  Blocking data: {len(blocking_df)} rows")
            if not blocking_df.empty:
                print(f"  Columns: {list(blocking_df.columns)}")
        except Exception as e:
            print(f"  Error fetching blocking: {e}")
            blocking_df = pd.DataFrame()

        if not blocking_df.empty:
            # Extract pass block grade column - PFF uses "PBLK" header
            pblk_cols = [c for c in blocking_df.columns if isinstance(c, str) and ('pblk' in c.lower() or ('pass' in c.lower() and 'block' in c.lower()))]
            if pblk_cols:
                blocking_df = blocking_df.rename(columns={pblk_cols[0]: 'pblk_grade'})

        # Scrape pass rush stats (PRSH)
        print("Fetching pass rush stats...")
        try:
            self.page.goto(self.PAGES['team_pass_rush'], wait_until='domcontentloaded', timeout=60000)
            time.sleep(5)
            self.screenshot('pff_passrush_page')
            rush_df = self.extract_pff_table()  # Use PFF-specific extraction
            print(f"  Pass rush data: {len(rush_df)} rows")
            if not rush_df.empty:
                print(f"  Columns: {list(rush_df.columns)}")
        except Exception as e:
            print(f"  Error fetching pass rush: {e}")
            rush_df = pd.DataFrame()

        if not rush_df.empty:
            # Extract pass rush grade column - PFF uses "PRSH" header
            prsh_cols = [c for c in rush_df.columns if isinstance(c, str) and ('prsh' in c.lower() or ('pass' in c.lower() and 'rush' in c.lower()))]
            if prsh_cols:
                rush_df = rush_df.rename(columns={prsh_cols[0]: 'prsh_grade'})

        # Merge data
        if not blocking_df.empty and not rush_df.empty:
            # Find team column - PFF uses "TEAM"
            team_col = None
            for col in blocking_df.columns:
                if isinstance(col, str) and 'team' in col.lower():
                    team_col = col
                    break
            if not team_col:
                # Use second column (index 1) which is typically team name
                team_col = blocking_df.columns[1] if len(blocking_df.columns) > 1 else blocking_df.columns[0]

            merged = blocking_df.merge(rush_df, on=team_col, how='outer', suffixes=('_blk', '_rush'))
            output_path = self.output_dir / "pff_team_stats.csv"
            merged.to_csv(output_path, index=False)
            print(f"Saved team stats to {output_path}")
            return merged
        elif not blocking_df.empty:
            output_path = self.output_dir / "pff_team_stats.csv"
            blocking_df.to_csv(output_path, index=False)
            print(f"Saved blocking stats only to {output_path}")
            return blocking_df

        return pd.DataFrame()

    def scrape_qb_stats(self) -> pd.DataFrame:
        """Scrape QB-level PFF stats (TWP, TWP%)"""
        print("\nFetching QB stats...")

        try:
            self.page.goto(self.PAGES['qb_passing'], wait_until='domcontentloaded', timeout=60000)
            time.sleep(5)
            self.screenshot('pff_qb_page')

            df = self.extract_pff_table()  # Use PFF-specific extraction
            print(f"  QB data: {len(df)} rows")
            if not df.empty:
                print(f"  Columns: {list(df.columns)}")

            if not df.empty:
                output_path = self.output_dir / "pff_qb_stats.csv"
                df.to_csv(output_path, index=False)
                print(f"Saved QB stats to {output_path}")

            return df
        except Exception as e:
            print(f"  Error fetching QB stats: {e}")
            return pd.DataFrame()

    def scrape_all(self) -> dict:
        """Scrape all PFF stats"""
        results = {
            'team_stats': self.scrape_team_stats(),
            'qb_stats': self.scrape_qb_stats()
        }
        return results


def main():
    parser = argparse.ArgumentParser(description='NFL Stats Scraper')
    parser.add_argument('--nextgen', action='store_true', help='Scrape Next Gen Stats (free)')
    parser.add_argument('--pff', action='store_true', help='Scrape PFF (requires subscription)')
    parser.add_argument('--all', action='store_true', help='Scrape all sources')
    parser.add_argument('-e', '--email', help='PFF account email')
    parser.add_argument('-p', '--password', help='PFF account password')
    parser.add_argument('-i', '--interactive', action='store_true', help='Visible browser')
    parser.add_argument('-s', '--season', type=int, default=2025, help='NFL season year (default: 2025)')

    args = parser.parse_args()

    # Default to nextgen if nothing specified
    if not args.nextgen and not args.pff and not args.all:
        args.nextgen = True

    headless = not args.interactive
    season = args.season

    print(f"\n*** Scraping {season} NFL Season Stats ***\n")

    # Scrape Next Gen Stats (free)
    if args.nextgen or args.all:
        print("\n" + "=" * 60)
        print(f"NEXT GEN STATS (FREE) - {season} SEASON")
        print("=" * 60)

        scraper = NextGenStatsScraper(headless=headless, season=season)
        try:
            scraper.start()
            scraper.scrape_all()
        finally:
            scraper.stop()

    # Scrape PFF (subscription)
    if args.pff or args.all:
        if not args.email:
            print("\nPFF requires email. Use: --email 'your@email.com'")
            return

        password = args.password
        if not password:
            import getpass
            password = getpass.getpass('PFF Password: ')

        print("\n" + "=" * 60)
        print(f"PFF STATS (SUBSCRIPTION) - {season} SEASON")
        print("=" * 60)

        scraper = PFFScraper(email=args.email, password=password, headless=headless, season=season)
        try:
            scraper.start()
            if scraper.login():
                scraper.scrape_all()
        finally:
            scraper.stop()

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {Path(__file__).parent.parent / 'data' / 'stats'}")


if __name__ == '__main__':
    main()
