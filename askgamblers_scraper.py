"""
AskGamblers Scraper Module
Scrapes casino reviews from AskGamblers to gather player experiences
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import re
from urllib.parse import quote


class AskGamblersScraper:
    """
    Scraper for extracting casino reviews from AskGamblers
    """

    BASE_URL = "https://www.askgamblers.com"
    SEARCH_URL = f"{BASE_URL}/search"

    def __init__(self, timeout: int = 10):
        """
        Initialize the scraper

        Args:
            timeout: Request timeout in seconds (default: 10)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'max-age=0',
            'Sec-Ch-Ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.google.com/'
        })

    def search_casino(self, casino_name: str) -> List[Dict]:
        """
        Search for a casino on AskGamblers by name

        Args:
            casino_name: Name of the casino (e.g., "stake", "bc.game")

        Returns:
            List of search results with their URLs and names
        """
        try:
            # Clean casino name for search (remove .com, .io, etc.)
            search_term = re.sub(r'\.(com|io|net|org|casino)$', '', casino_name.lower())

            results = []

            print(f"Searching AskGamblers for: {casino_name}")

            # Try constructing direct URL patterns (most reliable)
            possible_urls = [
                f"{self.BASE_URL}/online-casinos/reviews/{search_term}-casino-review",
                f"{self.BASE_URL}/online-casinos/reviews/{search_term}-review",
                f"{self.BASE_URL}/online-casinos/{search_term}-casino",
                f"{self.BASE_URL}/online-casinos/reviews/{search_term}",
            ]

            # Try each possible URL pattern
            for url in possible_urls:
                try:
                    print(f"Trying URL: {url}")
                    time.sleep(1)  # Small delay between attempts

                    response = self.session.get(url, timeout=self.timeout, allow_redirects=True)

                    if response.status_code == 200:
                        # Verify it's a valid casino page
                        soup = BeautifulSoup(response.content, 'html.parser')

                        # Check if page has review content
                        has_reviews = (
                            soup.find('div', class_=re.compile(r'.*review.*', re.IGNORECASE)) or
                            soup.find('article', class_=re.compile(r'.*review.*', re.IGNORECASE)) or
                            soup.find(string=re.compile(r'user.*review', re.IGNORECASE))
                        )

                        if has_reviews:
                            results.append({
                                'url': response.url,
                                'name': search_term,
                                'matches': True
                            })
                            print(f"✓ Found working URL: {response.url}")
                            return results
                    else:
                        print(f"✗ Status {response.status_code}")

                except requests.exceptions.RequestException as e:
                    print(f"✗ Request failed: {str(e)[:100]}")
                    continue
                except Exception as e:
                    print(f"✗ Error: {str(e)[:100]}")
                    continue

            # If all direct attempts fail, return empty
            if not results:
                print("Could not find casino page with direct URLs")
                print("Note: AskGamblers may have anti-scraping protection active")

            return results

        except Exception as e:
            print(f"Error searching for casino: {e}")
            return []

    def verify_casino_on_page(self, url: str, expected_name: str) -> bool:
        """
        Verify the casino name on the review page matches expected name

        Args:
            url: URL of the AskGamblers review page
            expected_name: Expected casino name

        Returns:
            True if casino matches, False otherwise
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for casino name in multiple places
            # 1. Check page title
            title = soup.find('title')
            if title and expected_name.lower() in title.get_text().lower():
                return True

            # 2. Check h1 heading
            h1 = soup.find('h1')
            if h1 and expected_name.lower() in h1.get_text().lower():
                return True

            # 3. Check meta tags
            meta_title = soup.find('meta', property='og:title')
            if meta_title:
                content = meta_title.get('content', '')
                if expected_name.lower() in content.lower():
                    return True

            # 4. Check URL
            if expected_name.lower() in url.lower():
                return True

            return False

        except Exception as e:
            print(f"Error verifying casino: {e}")
            return False

    def is_review_recent(self, date_str: str, months: int = 6) -> bool:
        """
        Check if a review date is within the specified months

        Args:
            date_str: Date string from review
            months: Number of months to check (default: 6)

        Returns:
            True if review is recent, False otherwise
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=months * 30)

            # Try to parse various date formats
            date_formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO 8601 with milliseconds
                '%Y-%m-%dT%H:%M:%SZ',      # ISO 8601 without milliseconds
                '%Y-%m-%d',
                '%B %d, %Y',
                '%b %d, %Y',
                '%d %B %Y',
                '%d %b %Y',
                '%d/%m/%Y',
                '%m/%d/%Y',
            ]

            # Handle relative dates
            lower_date = date_str.lower()
            if any(word in lower_date for word in ['today', 'yesterday', 'day ago', 'days ago']):
                return True
            if 'hour' in lower_date or 'minute' in lower_date:
                return True
            if 'week' in lower_date:
                weeks = int(re.search(r'\d+', date_str).group()) if re.search(r'\d+', date_str) else 1
                return weeks <= (months * 4)
            if 'month' in lower_date:
                mons = int(re.search(r'\d+', date_str).group()) if re.search(r'\d+', date_str) else 1
                return mons <= months

            # Try parsing exact dates
            review_date = None
            for fmt in date_formats:
                try:
                    review_date = datetime.strptime(date_str.strip(), fmt)
                    break
                except ValueError:
                    continue

            if review_date:
                return review_date >= cutoff_date

            return False

        except Exception as e:
            print(f"Error parsing date '{date_str}': {e}")
            return False

    def scrape_reviews_from_page(self, url: str, months: int = 6) -> Tuple[List[Dict], bool]:
        """
        Scrape reviews from a single AskGamblers page

        Args:
            url: URL of the AskGamblers review page
            months: Only include reviews from last N months

        Returns:
            Tuple of (list of reviews, whether last review is recent)
        """
        try:
            print(f"Scraping reviews from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            reviews = []

            # Try multiple selectors for review cards
            review_cards = soup.find_all('div', class_=re.compile(r'.*review.*card.*', re.IGNORECASE))

            if not review_cards:
                review_cards = soup.find_all('div', class_=re.compile(r'.*user.*review.*', re.IGNORECASE))

            if not review_cards:
                review_cards = soup.find_all('article', class_=re.compile(r'.*review.*', re.IGNORECASE))

            if not review_cards:
                # Try finding by data attributes
                review_cards = soup.find_all('div', attrs={'data-review': True})

            last_review_is_recent = False

            for idx, card in enumerate(review_cards):
                try:
                    # Extract date
                    date_elem = card.find('time')
                    if not date_elem:
                        date_elem = card.find('span', class_=re.compile(r'.*date.*', re.IGNORECASE))
                    if not date_elem:
                        date_elem = card.find('div', class_=re.compile(r'.*date.*', re.IGNORECASE))

                    date_str = date_elem.get('datetime') if date_elem and date_elem.get('datetime') else (
                        date_elem.get_text().strip() if date_elem else ''
                    )

                    if not date_str:
                        continue

                    is_recent = self.is_review_recent(date_str, months)

                    # Track if last review is recent
                    if idx == len(review_cards) - 1:
                        last_review_is_recent = is_recent

                    # Only include recent reviews
                    if not is_recent:
                        continue

                    # Extract rating (AskGamblers typically uses 1-5 scale)
                    rating = None
                    rating_elem = card.find('div', class_=re.compile(r'.*rating.*', re.IGNORECASE))
                    if not rating_elem:
                        rating_elem = card.find('span', class_=re.compile(r'.*rating.*', re.IGNORECASE))

                    if rating_elem:
                        # Look for rating value in various formats
                        rating_text = rating_elem.get_text()
                        rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?', rating_text)
                        if rating_match:
                            rating = float(rating_match.group(1))

                        # Check data attributes
                        if not rating:
                            for attr in ['data-rating', 'data-score', 'rating']:
                                if rating_elem.get(attr):
                                    try:
                                        rating = float(rating_elem.get(attr))
                                        break
                                    except:
                                        pass

                    # Extract title
                    title_elem = card.find('h3')
                    if not title_elem:
                        title_elem = card.find('h4')
                    if not title_elem:
                        title_elem = card.find('div', class_=re.compile(r'.*title.*', re.IGNORECASE))

                    title = title_elem.get_text().strip() if title_elem else ''

                    # Extract review text
                    text_elem = card.find('p', class_=re.compile(r'.*review.*text.*', re.IGNORECASE))
                    if not text_elem:
                        text_elem = card.find('div', class_=re.compile(r'.*review.*content.*', re.IGNORECASE))
                    if not text_elem:
                        # Fallback: find first substantial p tag
                        all_p = card.find_all('p')
                        for p in all_p:
                            p_text = p.get_text().strip()
                            if p_text and len(p_text) > 20:
                                text_elem = p
                                break

                    text = text_elem.get_text().strip() if text_elem else ''

                    # Extract author
                    author_elem = card.find('span', class_=re.compile(r'.*author.*', re.IGNORECASE))
                    if not author_elem:
                        author_elem = card.find('div', class_=re.compile(r'.*user.*name.*', re.IGNORECASE))
                    if not author_elem:
                        author_elem = card.find('span', class_=re.compile(r'.*user.*', re.IGNORECASE))

                    author = author_elem.get_text().strip() if author_elem else 'Anonymous'

                    # Only add review if it has substantial content
                    if text or title:
                        review_data = {
                            'date': date_str,
                            'rating': rating,
                            'title': title,
                            'text': text,
                            'author': author,
                            'full_content': f"{title} {text}".strip()
                        }

                        reviews.append(review_data)

                except Exception as e:
                    print(f"Error parsing review card: {e}")
                    continue

            print(f"Found {len(reviews)} recent reviews on this page")
            return reviews, last_review_is_recent

        except Exception as e:
            print(f"Error scraping page: {e}")
            return [], False

    def scrape_casino_reviews(
        self,
        casino_name: str,
        max_reviews: int = 50,
        months: int = 6
    ) -> Dict:
        """
        Scrape all recent reviews for a casino

        Args:
            casino_name: Name or domain of casino (e.g., "stake.com" or "stake")
            max_reviews: Maximum number of reviews to scrape (default: 50)
            months: Only include reviews from last N months (default: 6)

        Returns:
            Dictionary with reviews and metadata
        """
        print(f"\n{'='*60}")
        print(f"Starting AskGamblers scrape for: {casino_name}")
        print(f"{'='*60}\n")

        # Step 1: Search for casino
        search_results = self.search_casino(casino_name)

        if not search_results:
            print("No search results found")
            return {
                'casino_name': casino_name,
                'reviews': [],
                'total_count': 0,
                'error': 'No results found on AskGamblers'
            }

        # Step 2: Find correct result by verifying casino name
        correct_url = None
        clean_name = re.sub(r'\.(com|io|net|org|casino)$', '', casino_name.lower())

        for idx, result in enumerate(search_results):
            print(f"\nChecking result #{idx + 1}: {result['name']}")

            if self.verify_casino_on_page(result['url'], clean_name):
                print(f"✓ Casino verified for: {result['url']}")
                correct_url = result['url']
                break
            else:
                print(f"✗ Casino mismatch for: {result['url']}")

            # Add timeout between checks
            time.sleep(1)

        if not correct_url:
            print("\nNo matching casino found in search results")
            return {
                'casino_name': casino_name,
                'reviews': [],
                'total_count': 0,
                'error': 'Casino verification failed'
            }

        # Step 3: Scrape reviews from pages
        all_reviews = []
        page = 1

        while len(all_reviews) < max_reviews:
            # AskGamblers pagination might use different URL patterns
            page_url = correct_url if page == 1 else f"{correct_url}?page={page}"

            print(f"\nScraping page {page}...")
            reviews, last_is_recent = self.scrape_reviews_from_page(page_url, months)

            if not reviews:
                print("No more reviews found")
                break

            all_reviews.extend(reviews)

            # Check if we should continue to next page
            if page == 1 and last_is_recent and len(all_reviews) < max_reviews:
                print("Last review on first page is recent, checking next page...")
                page += 1
                time.sleep(2)  # Polite delay between page requests
            else:
                print("Stopping pagination")
                break

        # Limit to max_reviews
        all_reviews = all_reviews[:max_reviews]

        result = {
            'casino_name': casino_name,
            'reviews': all_reviews,
            'total_count': len(all_reviews),
            'askgamblers_url': correct_url,
            'scraped_at': datetime.now().isoformat()
        }

        print(f"\n{'='*60}")
        print(f"Scraping complete: {len(all_reviews)} reviews collected")
        print(f"{'='*60}\n")

        return result

    def analyze_withdrawal_mentions(self, reviews: List[Dict]) -> Dict:
        """
        Analyze reviews for withdrawal-related mentions

        Args:
            reviews: List of review dictionaries

        Returns:
            Dictionary with withdrawal analysis
        """
        withdrawal_keywords = [
            'withdrawal', 'withdraw', 'payout', 'cashout', 'cash out',
            'payment', 'pay out', 'pending', 'processing', 'waiting'
        ]

        time_keywords = [
            'instant', 'immediately', 'fast', 'quick', 'slow', 'delayed',
            'hours', 'days', 'weeks', 'minutes', 'never', 'still waiting'
        ]

        withdrawal_reviews = []
        positive_mentions = []
        negative_mentions = []

        for review in reviews:
            content = review.get('full_content', '').lower()

            # Check if review mentions withdrawals
            if any(keyword in content for keyword in withdrawal_keywords):
                withdrawal_reviews.append(review)

                # Categorize as positive or negative based on rating and keywords
                rating = review.get('rating', 0)

                # Handle None ratings
                if rating is None:
                    rating = 0

                is_positive = rating >= 4 or any(word in content for word in ['instant', 'fast', 'quick', 'immediately', 'smooth'])
                is_negative = rating <= 2 or any(word in content for word in ['slow', 'delayed', 'never', 'still waiting', 'pending'])

                if is_positive and not is_negative:
                    positive_mentions.append(review)
                elif is_negative:
                    negative_mentions.append(review)

        return {
            'total_withdrawal_mentions': len(withdrawal_reviews),
            'positive_count': len(positive_mentions),
            'negative_count': len(negative_mentions),
            'reviews': withdrawal_reviews,
            'positive_reviews': positive_mentions,
            'negative_reviews': negative_mentions,
            'has_sufficient_data': len(withdrawal_reviews) >= 10
        }

    def extract_player_experiences(self, reviews: List[Dict], section_keywords: Dict[str, List[str]]) -> Dict:
        """
        Extract player experiences by section based on keywords

        Args:
            reviews: List of review dictionaries
            section_keywords: Dictionary mapping section names to keyword lists

        Returns:
            Dictionary with experiences by section
        """
        experiences = {}

        for section, keywords in section_keywords.items():
            matching_reviews = []

            for review in reviews:
                content = review.get('full_content', '').lower()

                # Check if review mentions any keywords for this section
                if any(keyword.lower() in content for keyword in keywords):
                    matching_reviews.append({
                        'rating': review.get('rating'),
                        'date': review.get('date'),
                        'title': review.get('title'),
                        'text': review.get('text')
                    })

            experiences[section] = {
                'count': len(matching_reviews),
                'reviews': matching_reviews,
                'has_sufficient_data': len(matching_reviews) >= 10
            }

        return experiences


def format_review_summary(review_data: Dict, include_details: bool = True) -> str:
    """
    Format review data into a readable summary for AI consumption

    Args:
        review_data: Dictionary returned from scrape_casino_reviews
        include_details: Whether to include individual review details

    Returns:
        Formatted string summary
    """
    if not review_data or review_data.get('total_count', 0) == 0:
        return f"No recent reviews found on AskGamblers for {review_data.get('casino_name', 'this casino')}."

    summary_parts = []

    # Header
    summary_parts.append(f"AskGamblers Player Reviews for {review_data['casino_name']}")
    summary_parts.append(f"Total recent reviews: {review_data['total_count']}")
    summary_parts.append(f"Source: {review_data.get('askgamblers_url', 'N/A')}")
    summary_parts.append("")

    # Data sufficiency warning
    if review_data['total_count'] < 10:
        summary_parts.append("NOTE: Less than 10 reviews available. Take this data with a grain of salt.")
        summary_parts.append("")

    # Review details
    if include_details and review_data.get('reviews'):
        summary_parts.append("Player Comments:")
        for idx, review in enumerate(review_data['reviews'][:20], 1):  # Limit to 20 reviews
            rating_str = f"{review.get('rating', 'N/A')}" if review.get('rating') else 'N/A'
            summary_parts.append(f"\n{idx}. [{rating_str}/5] {review.get('date', 'N/A')}")
            if review.get('title'):
                summary_parts.append(f"   Title: {review['title']}")
            if review.get('text'):
                text_preview = review['text'][:300]
                if len(review['text']) > 300:
                    text_preview += "..."
                summary_parts.append(f"   Review: {text_preview}")

    return "\n".join(summary_parts)
