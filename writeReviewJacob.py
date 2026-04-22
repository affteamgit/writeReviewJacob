import os
import openai
import requests
import json
import streamlit as st
from google.oauth2.service_account import Credentials   
from googleapiclient.discovery import build
from anthropic import Anthropic
from pathlib import Path
import re
import random
import concurrent.futures
from typing import Dict, Tuple
from askgamblers_scraper import AskGamblersScraper
from trustpilot_scraper import TrustpilotScraper

# CONFIG 
OPENAI_API_KEY      = st.secrets["OPENAI_API_KEY"]
GROK_API_KEY        = st.secrets["GROK_API_KEY"]
ANTHROPIC_API_KEY   = st.secrets["ANTHROPIC_API_KEY"]
COINMARKETCAP_API_KEY = st.secrets["COINMARKETCAP_API_KEY"]

SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]
SHEET_NAME     = st.secrets["SHEET_NAME"]

FOLDER_ID = st.secrets["FOLDER_ID"]
GUIDELINES_FOLDER_ID = st.secrets["GUIDELINES_FOLDER_ID"]

# Evolution comparison config
CALCULATION_SPREADSHEET_ID = "1av0ZgZQGPWErmlzFmCIyZg1ApkzOQPht2AUoB_MGvLg"

# AFF SITES spreadsheet for review dates
AFF_SITES_SPREADSHEET_ID = "1s7FcUQN57SnQ3Ihq2iewoPf5TpivX4PYmCo8zQCIocc"
AFF_SITES_TAB = "BCK"

# No fine-tuned model needed - Jacob will generate factual reviews only

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive"
]

DOCS_DRIVE_SCOPES = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]

def get_service_account_credentials():
    return Credentials.from_service_account_info(st.secrets["service_account"], scopes=SCOPES)

def get_file_content_from_github(filename):
    """Get content of a file from GitHub repository."""
    try:
        github_base_url = "https://raw.githubusercontent.com/affteamgit/writeReviewJacob/main/templates/"
        file_url = f"{github_base_url}{filename}.txt"
        
        response = requests.get(file_url)
        response.raise_for_status()
        
        return response.text
        
    except Exception as e:
        print(f"Error reading file {filename} from GitHub: {str(e)}")
        return None

def get_json_from_github(filename):
    """Get JSON content from GitHub repository and parse it."""
    try:
        github_base_url = "https://raw.githubusercontent.com/affteamgit/writeReviewJacob/main/templates/"
        file_url = f"{github_base_url}{filename}.json"
        response = requests.get(file_url)
        response.raise_for_status()
        return json.loads(response.text)
    except Exception as e:
        print(f"Error reading JSON file {filename} from GitHub: {str(e)}")
        return None


def get_all_templates():
    """Fetch all templates at once with parallel processing"""
    templates = {}
    files = [
        'PromptTemplate',
        'BaseGuidelinesClaude',
        'BaseGuidelinesGrok',
        'StructureTemplateGeneral',
        'StructureTemplatePayments',
        'StructureTemplateGames',
        'StructureTemplateResponsible',
        'StructureTemplateFAQ'
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(get_file_content_from_github, filename): filename
                         for filename in files}
        # Also fetch personalization matrix as JSON
        persona_future = executor.submit(get_json_from_github, 'PersonalizationMatrix')

        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                templates[filename] = future.result()
            except Exception as e:
                print(f"Error fetching template {filename}: {e}")
                templates[filename] = None

        # Collect personalization matrix
        try:
            templates["PersonalizationMatrix"] = persona_future.result()
        except Exception as e:
            print(f"Error fetching PersonalizationMatrix: {e}")
            templates["PersonalizationMatrix"] = None

    return templates

def get_selected_casino_data():
    creds = get_service_account_credentials()
    sheets = build("sheets", "v4", credentials=creds)
    casino = sheets.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=f"{SHEET_NAME}!B1").execute().get("values", [[""]])[0][0].strip()
    rows = sheets.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=f"{SHEET_NAME}!B2:S").execute().get("values", [])
    sections = {
        "General": (2, 3, 4),
        "Payments": (5, 6, 7),
        "Games": (8, 9, 10),
        "Responsible Gambling": (11, 12, 13),
        "Bonuses": (14, 15, 16),
    }
    data = {}
    comments_column = 17  # Column S (0-indexed)
    
    # Extract comments from column S
    all_comments = "\n".join(r[comments_column] for r in rows if len(r) > comments_column and r[comments_column].strip())
    
    for sec, (mi, ti, si) in sections.items():
        main = "\n".join(r[mi] for r in rows if len(r) > mi and r[mi].strip())
        if ti is not None:
            top = "\n".join(r[ti] for r in rows if len(r) > ti and r[ti].strip())
        else:
            top = "[No top comparison available]"
        if si is not None:
            sim = "\n".join(r[si] for r in rows if len(r) > si and r[si].strip())
        else:
            sim = "[No similar comparison available]"
        data[sec] = {"main": main or "[No data provided]", "top": top, "sim": sim}

    # Casino ID is in B2 (first row, first column of the B2:S range)
    casino_id = rows[0][0].strip() if rows and len(rows[0]) > 0 else None

    return casino, data, all_comments, casino_id

def get_cached_casino_data():
    """Get casino data without caching to prevent tone interference"""
    return get_selected_casino_data()

def get_review_wp_id(casino_id):
    """Look up the WordPress post ID for a casino's previous review.
    Reads the WP_IDs tab from the calculation spreadsheet.
    Column A = internal Casino ID, Column B = review_WP_ID.
    """
    if not casino_id or not CALCULATION_SPREADSHEET_ID:
        return None
    try:
        creds = get_service_account_credentials()
        sheets = build("sheets", "v4", credentials=creds)
        result = sheets.spreadsheets().values().get(
            spreadsheetId=CALCULATION_SPREADSHEET_ID,
            range="WP_IDs!A:B"
        ).execute()
        rows = result.get("values", [])
        for row in rows:
            if len(row) >= 2 and row[0].strip() == casino_id:
                return row[1].strip()
        return None
    except Exception as e:
        print(f"Error looking up WP ID for casino {casino_id}: {e}")
        return None

def fetch_old_review_from_mysql(wp_id):
    """Fetch post_content and post_name from WordPress wp_posts for a given post ID.
    Returns (content, post_name) tuple, or (None, None) on failure."""
    if not wp_id:
        return None, None
    try:
        import pymysql
        mysql_config = st.secrets.get("mysql", {})
        if not mysql_config:
            print("MySQL secrets not configured, skipping old review fetch")
            return None, None
        connection = pymysql.connect(
            host=mysql_config["host"],
            port=int(mysql_config.get("port", 3306)),
            user=mysql_config["user"],
            password=mysql_config["password"],
            database=mysql_config["database"],
            connect_timeout=10,
            read_timeout=15
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT post_content, post_name FROM wp_posts WHERE ID = %s", (int(wp_id),))
                result = cursor.fetchone()
                if result:
                    return result[0], result[1]
                return None, None
        finally:
            connection.close()
    except Exception as e:
        print(f"Error fetching old review from MySQL (WP ID {wp_id}): {e}")
        return None, None

def strip_html_to_text(html_content):
    """Strip WordPress HTML to plain text for AI processing."""
    if not html_content:
        return ""
    text = re.sub(r'<[^>]+>', ' ', html_content)
    text = re.sub(r'\[/?[^\]]+\]', ' ', text)  # WordPress shortcodes
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ').replace('&#8217;', "'").replace('&#8216;', "'")
    text = text.replace('&#8220;', '"').replace('&#8221;', '"')
    return text

def extract_evolution_facts(casino_name, old_review_text):
    """Use Claude to extract key comparable facts from an old review, organized by section."""
    if not old_review_text or len(old_review_text.strip()) < 100:
        return {}

    # Truncate very long reviews to stay within reasonable token limits
    max_chars = 15000
    if len(old_review_text) > max_chars:
        old_review_text = old_review_text[:max_chars]

    prompt = f"""Analyze this previous review of the casino "{casino_name}" and extract key factual data points organized by section.

For each section, extract ONLY concrete, measurable facts that could be compared to current data. Focus on:
- Specific numbers (game counts, provider counts, crypto counts, withdrawal limits, bonus amounts)
- Specific features that were present or absent (VPN policy, KYC requirements, self-exclusion tools)
- Specific timeframes (withdrawal processing times, KYC verification times)
- Specific bonus terms (wagering requirements, max cashout)

Do NOT extract opinions, subjective assessments, or writing style. Only extract verifiable facts.

Output ONLY valid JSON with this exact structure (no markdown, no code fences):
{{"General": "bullet points of facts or empty string", "Payments": "bullet points of facts or empty string", "Games": "bullet points of facts or empty string", "Responsible Gambling": "bullet points of facts or empty string", "Bonuses": "bullet points of facts or empty string"}}

If a section has no relevant facts in the old review, use an empty string for that section.

Previous review text:
{old_review_text}"""

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text.strip()

        # Strip markdown code fences if the AI wrapped the JSON
        cleaned = response
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

        # Extract just the JSON object if AI appended extra text after it
        brace_start = cleaned.find('{')
        if brace_start != -1:
            depth = 0
            for i, ch in enumerate(cleaned[brace_start:], brace_start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        cleaned = cleaned[brace_start:i + 1]
                        break

        facts = json.loads(cleaned)
        valid_sections = {"General", "Payments", "Games", "Responsible Gambling", "Bonuses"}
        return {k: v for k, v in facts.items() if k in valid_sections and isinstance(v, str)}
    except json.JSONDecodeError as e:
        print(f"Error parsing evolution facts JSON: {e}")
        print(f"Raw AI response (first 500 chars): {response[:500]}")
        return {}
    except Exception as e:
        print(f"Error extracting evolution facts: {e}")
        return {}

def _compute_relative_time(date_value):
    """Convert a date into a human-readable relative time phrase.
    Accepts datetime objects or strings in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format."""
    if not date_value:
        return None
    try:
        from datetime import datetime
        if isinstance(date_value, str):
            date_value = date_value.strip()
            # Try YMD format first (from AFF SITES spreadsheet)
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"]:
                try:
                    date_value = datetime.strptime(date_value, fmt)
                    break
                except ValueError:
                    continue
            else:
                print(f"Could not parse date string: {date_value}")
                return None
        now = datetime.now()
        delta = now - date_value
        months = delta.days // 30
        if months < 1:
            return "a few weeks ago"
        elif months == 1:
            return "about a month ago"
        elif months <= 3:
            return "a couple of months ago"
        elif months <= 6:
            return "a few months ago"
        elif months <= 11:
            return "about half a year ago"
        elif months <= 14:
            return "about a year ago"
        elif months <= 20:
            return "over a year ago"
        elif months <= 30:
            return "about two years ago"
        else:
            years = months // 12
            return f"about {years} years ago"
    except Exception as e:
        print(f"Error computing relative time: {e}")
        return None

def _get_review_date_from_affsites(post_name):
    """Look up the Last Updated date from the AFF SITES spreadsheet.
    Builds /reviews/{post_name}/ slug and matches against Column D (URL Slug).
    Returns the Last Updated date string from Column E, or None."""
    if not post_name:
        return None
    try:
        creds = get_service_account_credentials()
        sheets = build("sheets", "v4", credentials=creds)
        # Read columns D (URL Slug) and E (Last Updated) from BCK tab
        result = sheets.spreadsheets().values().get(
            spreadsheetId=AFF_SITES_SPREADSHEET_ID,
            range=f"{AFF_SITES_TAB}!D:E"
        ).execute()
        rows = result.get("values", [])

        # Build the slug to match: /reviews/post_name/
        target_slug = f"/reviews/{post_name}/"
        # Also try without trailing slash and with variations
        target_variants = [
            target_slug,
            f"/reviews/{post_name}",
            f"/{post_name}/",
            f"/{post_name}",
        ]

        for row in rows:
            if not row or not row[0]:
                continue
            slug = row[0].strip()
            if slug in target_variants:
                if len(row) >= 2 and row[1].strip():
                    date_str = row[1].strip()
                    print(f"Found review date in AFF SITES: slug={slug}, date={date_str}")
                    return date_str
                else:
                    print(f"Found slug {slug} in AFF SITES but no date in column E")
                    return None

        print(f"No matching slug found in AFF SITES for post_name={post_name} (tried {target_variants[0]})")
        return None
    except Exception as e:
        print(f"Error looking up review date from AFF SITES: {e}")
        return None

def fetch_and_extract_evolution_data(casino_id, casino_name):
    """Full pipeline: look up WP ID -> fetch old review -> get review date from AFF SITES -> extract facts.
    Returns (facts_dict, status_message) tuple.
    facts_dict may include a special "_relative_time" key with the time since the old review.
    """
    if not casino_id:
        return {}, "No Casino ID found -- old review comparison skipped"

    wp_id = get_review_wp_id(casino_id)
    if not wp_id:
        return {}, f"No WP_ID found in spreadsheet for Casino ID {casino_id}"
    print(f"Found WP ID {wp_id} for casino ID {casino_id}")

    html_content, post_name = fetch_old_review_from_mysql(wp_id)
    if html_content is None:
        return {}, f"Could not fetch review from database for WP ID {wp_id} (connection issue or post not found)"

    # Get the review date from AFF SITES spreadsheet using post_name slug
    review_date_str = _get_review_date_from_affsites(post_name)
    relative_time = _compute_relative_time(review_date_str)
    if relative_time:
        print(f"Review date from AFF SITES: {review_date_str} ({relative_time})")
    elif post_name:
        print(f"Could not determine review date for post_name={post_name}")

    plain_text = strip_html_to_text(html_content)
    if not plain_text or len(plain_text.strip()) < 100:
        return {}, f"Old review text too short after HTML stripping ({len(plain_text)} chars) for WP ID {wp_id}"
    print(f"Fetched old review: {len(plain_text)} chars")

    facts = extract_evolution_facts(casino_name, plain_text)

    # Attach relative time to facts so it can be included in the prompt
    if relative_time:
        facts["_relative_time"] = relative_time

    non_empty = {k: v for k, v in facts.items() if v.strip() and not k.startswith("_")}
    if non_empty:
        return facts, f"Old review data integrated ({len(non_empty)} section{'s' if len(non_empty) != 1 else ''} with evolution info, reviewed {relative_time or 'unknown time ago'})"
    else:
        return facts, f"Old review fetched (WP ID {wp_id}, {len(plain_text)} chars) but no comparable facts extracted"

# AI CLIENTS
client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

def call_openai(prompt):
    # Add fact constraint system message
    fact_constraint = "CRITICAL: Only use facts explicitly provided in the prompt. Never add information not in the source data. Do not make assumptions or add general knowledge about casinos. Never claim exclusivity or uniqueness unless the data explicitly states it."
    full_prompt = f"{fact_constraint}\n\n{prompt}"
    return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": full_prompt}], temperature=0.45, max_tokens=1200).choices[0].message.content.strip()

def call_grok(prompt):
    # Add fact constraint system message
    fact_constraint = "CRITICAL: Only use facts explicitly provided in the prompt. Never add information not in the source data. Do not make assumptions or add general knowledge about casinos. Never claim exclusivity or uniqueness unless the data explicitly states it."
    full_prompt = f"{fact_constraint}\n\n{prompt}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK_API_KEY}"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": full_prompt}], "temperature": 0.45, "max_tokens": 1200}
    j = requests.post("https://api.x.ai/v1/chat/completions", json=payload, headers=headers).json()
    return j.get("choices", [{}])[0].get("message", {}).get("content", "[Grok failed]").strip()

def call_claude(prompt):
    # Add fact constraint system message
    fact_constraint = "CRITICAL: Only use facts explicitly provided in the prompt. Never add information not in the source data. Do not make assumptions or add general knowledge about casinos. Never claim exclusivity or uniqueness unless the data explicitly states it."
    full_prompt = f"{fact_constraint}\n\n{prompt}"
    return anthropic_client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1200, temperature=0.45, messages=[{"role": "user", "content": full_prompt}]).content[0].text.strip()

def get_casino_reputation_summary(casino_name: str) -> str:
    """Use Claude to generate a reputation summary of the casino based on its general knowledge.

    This is intentionally NOT constrained to spreadsheet data -- the AI uses its training
    knowledge to provide a reputation overview for the 'Is [casino] legit?' question.
    """
    prompt = f"""Provide a brief factual reputation summary of the crypto casino "{casino_name}". Include:
- When it was founded/launched (if known)
- Licensing information (e.g., Curacao, Malta Gaming Authority, etc.)
- Any notable reputation milestones (awards, partnerships)
- Any notable controversies or negative incidents
- General community standing among crypto gambling players

Keep it to 3-5 sentences. Be factual and balanced. If you don't have reliable information about this casino, say so clearly -- do not fabricate details.

Output ONLY the summary paragraph, no headings or labels."""

    try:
        # Call Claude WITHOUT the strict fact constraint -- we want general knowledge here
        result = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text.strip()
        return result
    except Exception as e:
        print(f"Reputation summary failed for {casino_name}: {e}")
        return "No reputation data available."

def extract_casino_names_from_data(comparison_data):
    """Extract casino names from comparison data string.
    Assumes format like 'CasinoName (link): data...' or '[CasinoName](link): data...'
    """
    casino_names = []
    # Match patterns like:
    # - "CasinoName (https://...)"
    # - "[CasinoName](https://...)"
    # - "CasinoName:"
    patterns = [
        r'\[([^\]]+)\]\(https?://[^\)]+\)',  # [CasinoName](link)
        r'^([A-Z][A-Za-z0-9\s\.]+?)(?:\s*\(https?://|\s*:)',  # CasinoName (link or :
    ]

    for line in comparison_data.split('\n'):
        line = line.strip()
        if not line or line.startswith('[No '):
            continue
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                casino_name = match.group(1).strip()
                if casino_name and casino_name not in casino_names:
                    casino_names.append(casino_name)
                break

    return casino_names

# Removed extract_casino_links_map - not needed for factual Jacob reviews

def get_next_comparison_casino(available_casinos, used_casinos_tracker):
    """Select next casino using shuffled round-robin logic.

    Args:
        available_casinos: List of casino names available for comparison
        used_casinos_tracker: List tracking recently used casinos (max 5)

    Returns:
        Selected casino name or None if no casinos available
    """
    if not available_casinos:
        return None

    # Filter out recently used casinos (within last 5 uses)
    available = [c for c in available_casinos if c not in used_casinos_tracker[-5:]]

    # If all casinos have been used recently, reset with full list
    if not available:
        available = list(available_casinos)

    # Shuffle to avoid always picking the same first casino
    shuffled = list(available)
    random.shuffle(shuffled)
    return shuffled[0] if shuffled else None

def update_used_casinos_tracker(tracker, casino_name):
    """Add casino to the used tracker list."""
    if casino_name:
        tracker.append(casino_name)
    return tracker

def sort_comments_by_section(comments):
    """Use AI to intelligently sort comments by section."""
    if not comments or not comments.strip():
        return {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "FAQ": ""}

    prompt = f"""Please analyze the following feedback comments and sort them by the casino review sections they belong to.

Comments:
{comments}

Sections:
- General (unique casino features, standout features, bonuses, welcome offers, promotions, VIP programs, loyalty rewards, gamification, ongoing promos, reputation, establishment date, etc.)
- Payments (deposits, withdrawals, KYC, payment methods, processing times, withdrawal restrictions, etc.)
- Games (game selection, slots, table games, live casino, game providers, etc.)
- Responsible Gambling (limits, self-exclusion, problem gambling tools, etc.)
- FAQ (VPN support, anonymity, registration process, customer support quality, highroller features, casino ownership, etc.)

For each section, return ONLY the comments that belong to that section. If no comments belong to a section, leave it empty.

Format your response exactly like this:
**General**
[relevant comments here or leave empty]

**Payments**
[relevant comments here or leave empty]

**Games**
[relevant comments here or leave empty]

**Responsible Gambling**
[relevant comments here or leave empty]

**FAQ**
[relevant comments here or leave empty]"""

    try:
        response = call_claude(prompt)
        # Parse the response into a dictionary
        sections = {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "FAQ": ""}
        current_section = None

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('**') and line.endswith('**'):
                section_name = line[2:-2]  # Remove ** from both ends
                if section_name in sections:
                    current_section = section_name
            elif current_section and line:
                if sections[current_section]:
                    sections[current_section] += " " + line
                else:
                    sections[current_section] = line

        return sections
    except Exception as e:
        print(f"Error sorting comments: {e}")
        # Fallback: return empty sections
        return {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "FAQ": ""}

def parse_review_sections(content):
    """Parse review content into sections based on **Section Name** format."""
    section_headers = [
        "General",
        "Payments", 
        "Games",
        "Responsible Gambling",
        "Bonuses"
    ]
    
    lines = content.split('\n')
    sections = []
    current_section = None
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if this line is a section header in **Section Name** format
        is_header = False
        for header in section_headers:
            if line_stripped == f"**{header}**":
                # Save previous section if exists
                if current_section and current_content:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content).strip()
                    })
                
                # Start new section
                current_section = header
                current_content = []
                is_header = True
                break
        
        # If not a header, add to current content
        if not is_header:
            if current_section is None:
                # Skip content before the first section header
                continue
            current_content.append(line)
    
    # Don't forget the last section
    if current_section and current_content:
        sections.append({
            'title': current_section,
            'content': '\n'.join(current_content).strip()
        })
    
    return sections

# Removed add_internal_links_to_casinos - not needed for factual Jacob reviews

def scrape_player_feedback(casino_name: str) -> dict:
    """Scrape AskGamblers and Trustpilot for player feedback about a casino.

    Tries both sources and merges the reviews. Returns a dict with raw reviews
    and source info, or empty dict if nothing found.
    """
    all_reviews = []
    sources = []

    # --- AskGamblers ---
    try:
        ag_scraper = AskGamblersScraper(timeout=30)
        ag_data = ag_scraper.scrape_casino_reviews(casino_name, max_reviews=50, months=6)
        ag_reviews = ag_data.get('reviews', [])
        if ag_reviews:
            all_reviews.extend(ag_reviews)
            sources.append("AskGamblers")
            print(f"AskGamblers: {len(ag_reviews)} reviews")
    except Exception as e:
        import traceback
        print(f"AskGamblers scrape failed for {casino_name}: {e}")
        traceback.print_exc()

    # --- Trustpilot ---
    try:
        tp_scraper = TrustpilotScraper(timeout=30)
        tp_data = tp_scraper.scrape_casino_reviews(casino_name, max_reviews=50, months=6)
        tp_reviews = tp_data.get('reviews', [])
        if tp_reviews:
            all_reviews.extend(tp_reviews)
            sources.append("Trustpilot")
            print(f"Trustpilot: {len(tp_reviews)} reviews")
    except Exception as e:
        import traceback
        print(f"Trustpilot scrape failed for {casino_name}: {e}")
        traceback.print_exc()

    if not all_reviews:
        return {}

    source_str = " and ".join(sources)
    return {
        "casino_name": casino_name,
        "reviews": all_reviews,
        "total_count": len(all_reviews),
        "source_str": source_str,
    }


def _prepare_reviews_for_prompt(reviews, max_reviews=25):
    """Prepare a compact text block of reviews for an AI prompt."""
    lines = []
    for i, rev in enumerate(reviews[:max_reviews], 1):
        rating = rev.get('rating', 'N/A')
        text = (rev.get('text') or rev.get('title') or '').strip()
        if not text:
            continue
        # Truncate long reviews but keep enough for specifics
        if len(text) > 500:
            text = text[:500].rsplit(' ', 1)[0] + "..."
        lines.append(f"[{rating}] {text}")
    return "\n".join(lines)


def generate_general_player_summary(casino_name, feedback_data):
    """Use Claude to summarize player reviews for the General section."""
    reviews = feedback_data["reviews"]
    source_str = feedback_data["source_str"]
    total = feedback_data["total_count"]

    reviews_text = _prepare_reviews_for_prompt(reviews)
    if not reviews_text:
        return ""

    # Pick a random style to force structural variation
    styles = [
        {
            "instruction": "Lead with the most specific, concrete thing players keep bringing up, then add Jakob's take.",
            "example": f"Players keep coming back to {casino_name}'s **instant crypto withdrawals**, and after testing it myself I can confirm my BTC hit my wallet in under 10 minutes."
        },
        {
            "instruction": "Start with Jakob's own reaction to what he found in the reviews, then cite specific details.",
            "example": f"I went through player feedback for {casino_name} and one thing kept coming up: the **support team actually resolves issues fast**. Multiple players described getting stuck withdrawals sorted within hours, not days."
        },
        {
            "instruction": "Start with the dominant theme across reviews (positive or negative), backed by specifics.",
            "example": f"The recurring theme in {casino_name}'s player feedback is **speed**. Fast deposits, fast withdrawals, fast support responses. Players specifically mention getting payouts processed in under 2 hours."
        },
        {
            "instruction": "Lead with something surprising or notable from the reviews that stood out to Jakob.",
            "example": f"The thing that stood out in player feedback is that {casino_name} players rarely complain about withdrawal delays. Several describe getting **paid out the same day**, which is not something I see often."
        },
    ]
    style = random.choice(styles)

    prompt = f"""You are Jakob, a crypto casino reviewer. Summarize what players say about {casino_name} based on the reviews below. Write this as Jakob reacting to what he found in the feedback.

STYLE: {style["instruction"]}
Example (adapt, don't copy): {style["example"]}

APPROACH:
- Be SPECIFIC. Pull out concrete details from the reviews: specific features praised, specific problems described, actual experiences mentioned. Vague phrases like "some complaints emerge" or "players are generally satisfied" are useless. What EXACTLY are they satisfied about? What EXACTLY are they complaining about?
- Cite the substance of what players say without quoting them directly. For example: "Players describe getting withdrawals processed in under 2 hours" or "the most common complaint is account verification taking over a week" or "several players mention the live chat resolving issues on the spot."
- Add Jakob's own voice. React to what you found: does it match your own experience testing this casino? Does something surprise you? Do the complaints seem legitimate or like sore losers?
- If a specific complaint or praise appears multiple times across reviews, that pattern matters more than one-off mentions.

RULES:
- CRITICAL: NEVER mention Trustpilot, AskGamblers, or any review platform by name. Say "player reviews" or "player feedback" instead. This is a hard requirement.
- NEVER use the word "platform" to refer to a casino. Use "casino" or "site" instead.
- 70-120 words total. Packed with specifics, not padded with filler.
- Break into 2-3 SHORT paragraphs (2-3 sentences each) separated by blank lines. Do NOT write one long wall of text. For example: first paragraph covers the positives, second paragraph covers the negatives or concerns.
- FORBIDDEN phrases: "Based on recent reviews", "players frequently praise", "players consistently praise", "with many highlighting", "some complaints emerge", "players report mixed experiences", "feedback is generally positive". These are all too vague.
- Do NOT mention how many reviews were analyzed.
- Weigh positives more heavily. Negative reviews often come from emotional post-loss players. But if a real pattern of complaints exists, mention the specific issue.
- NEVER use em dashes (—), en dashes (–), or hyphens as clause connectors. Use commas, periods, or rewrite the sentence. This is a hard formatting requirement.
- Use "I" and "you". Bold key points with **asterisks**.
- ALWAYS use commas for numbers over 999: $1,500 not $1500.
- Output ONLY the paragraph text, no heading.

REVIEWS:
{reviews_text}"""

    try:
        summary = call_claude(prompt)
        result = f"\n## Q: What do players say about {casino_name}?\n\n{summary}"
        return result
    except Exception as e:
        print(f"General player summary AI call failed: {e}")
        return ""


def generate_withdrawal_player_summary(casino_name, feedback_data):
    """Use Claude to summarize withdrawal-related player feedback for the Payments section."""
    reviews = feedback_data["reviews"]

    reviews_text = _prepare_reviews_for_prompt(reviews, max_reviews=30)
    if not reviews_text:
        return ""

    prompt = f"""From the following player reviews about {casino_name}, extract and summarize ONLY the feedback related to withdrawals (cashouts, payouts, cashing out, withdrawal speed, withdrawal problems, pending withdrawals, declined withdrawals).

RULES:
- CRITICAL: NEVER mention Trustpilot, AskGamblers, or any review platform by name. Use "player reviews", "player feedback", or "what players report" instead.
- NEVER use the word "platform" to refer to a casino. Use "casino" or "site" instead.
- NEVER single out individual reviews (e.g., "one player noted"). Always generalize: "some players report", "players mention", "feedback suggests".
- If NO reviews mention withdrawals at all, respond with exactly: NO_WITHDRAWAL_FEEDBACK
- One short paragraph, 40-70 words. Direct, no filler.
- Focus on whether players report withdrawal problems or smooth experiences.
- Be specific about the issues if any: delays, declined withdrawals, stuck pending status, etc.
- If feedback is mostly positive about withdrawals, say so clearly.
- NEVER use em dashes (—), en dashes (–), or hyphens as clause connectors. Use commas, periods, or rewrite the sentence.
- Use "I" and "you". Bold key points with **asterisks**.
- ALWAYS use commas for numbers over 999: $1,500 not $1500.
- Output ONLY the paragraph text, no heading.

REVIEWS:
{reviews_text}"""

    try:
        summary = call_claude(prompt)
        if not summary or "NO_WITHDRAWAL_FEEDBACK" in summary:
            return ""
        return summary
    except Exception as e:
        print(f"Withdrawal player summary AI call failed: {e}")
        return ""


def select_persona_scenario(section: str, main_data: str, matrix: dict, used_ids_this_review: list) -> dict:
    """Select a random personalization scenario for a section based on casino data."""
    section_hooks = matrix.get("sections", {}).get(section, [])
    if not section_hooks:
        return None

    main_lower = main_data.lower()

    # Filter to hooks whose data_keywords match the casino data
    eligible = []
    fallbacks = []
    for hook in section_hooks:
        keywords = hook.get("data_keywords", [])
        is_fallback = hook.get("fallback_ok", False)

        # Skip if already used in this review
        if hook["id"] in used_ids_this_review:
            continue

        # If no keywords, it's a fallback-only hook
        if not keywords:
            if is_fallback:
                fallbacks.append(hook)
            continue

        # Check if at least one keyword matches
        if any(kw.lower() in main_lower for kw in keywords):
            eligible.append(hook)
        elif is_fallback:
            fallbacks.append(hook)

    # Pick from eligible first, then fallbacks
    if eligible:
        return random.choice(eligible)
    if fallbacks:
        return random.choice(fallbacks)
    return None


def build_personalization_block(scenario: dict) -> str:
    """Build the personalization instruction block to inject into the prompt."""
    instruction = scenario.get("instruction", "")

    # Randomized framing layer -- changes HOW the personalization reads,
    # so even the same scenario produces different narrative structures
    framings = [
        "FRAMING: Start the answer mid-action, as if the reader joins Jakob in the middle of testing this casino. No preamble.",
        "FRAMING: Open with what surprised or caught Jakob's attention about this specific aspect. Lead with the unexpected.",
        "FRAMING: Lead with Jakob's verdict first, then walk the reader through how he got there.",
        "FRAMING: Frame this as if Jakob is telling a friend who asked him specifically about this. Conversational, direct.",
        "FRAMING: Start with a specific detail Jakob noticed, then zoom out to the bigger picture.",
        "FRAMING: Contrast this casino with what Jakob usually sees. What's different here -- better or worse?",
        "FRAMING: Write this as a sequence of what Jakob did -- first he checked X, then he tried Y, and here's what he found.",
        "FRAMING: Start with the casino data point that matters most for this question, then layer in Jakob's reaction to it.",
    ]
    framing = random.choice(framings)

    return (
        f"\n\nPERSONAL EXPERIENCE INSTRUCTION (Jakob's personal touch):\n"
        f"{instruction}\n"
        f"{framing}\n"
        f"CRITICAL: REWRITE the targeted answer from Jakob's experience -- narrate his actions "
        f"at this casino. Do NOT write a factual answer and bolt a personal sentence onto it. "
        f"The answer itself should read as Jakob's experience, with data points woven into what "
        f"he did and encountered. Every fact, feature, and number must still come from the casino "
        f"data below. Never invent features, game names, or capabilities not in the data."
    )


def generate_sections_parallel(casino: str, secs: Dict, sorted_comments: Dict, templates: Dict, btc_str: str, evolution_facts: Dict[str, str] = None) -> list:
    """Generate all sections in parallel while maintaining round-robin casino selection"""

    # Initialize tracker for used comparison casinos
    used_casinos_tracker = []

    # Pre-assign a rotation list of casinos to each section
    # We need to do this sequentially before parallel generation
    section_order = ["General", "Payments", "Games", "Responsible Gambling", "FAQ"]
    section_assignments = {}

    for sec in section_order:
        if sec in secs:
            content = secs[sec]
            # Extract available casinos
            top_casinos = extract_casino_names_from_data(content["top"])
            sim_casinos = extract_casino_names_from_data(content["sim"])
            all_available = top_casinos + sim_casinos

            # Build a prioritized rotation list for this section
            # This list will help the AI rotate through different casinos for different comparisons
            rotation_list = []
            temp_tracker = used_casinos_tracker.copy()

            # Get up to 4 casinos for rotation within this section
            for _ in range(min(4, len(all_available))):
                next_casino = get_next_comparison_casino(all_available, temp_tracker)
                if next_casino:
                    rotation_list.append(next_casino)
                    temp_tracker.append(next_casino)

            # Update the main tracker with the first casino from this section's rotation
            if rotation_list:
                used_casinos_tracker.append(rotation_list[0])

            # Store assignment
            section_assignments[sec] = rotation_list

    # Pre-select personalization scenarios sequentially (before parallel generation)
    personalization_matrix = templates.get("PersonalizationMatrix")
    section_persona = {}
    used_scenario_ids = []
    if personalization_matrix:
        for sec in section_order:
            if sec in secs:
                scenario = select_persona_scenario(sec, secs[sec]["main"], personalization_matrix, used_scenario_ids)
                if scenario:
                    section_persona[sec] = scenario
                    used_scenario_ids.append(scenario["id"])
                    print(f"Section {sec}: Persona scenario = {scenario['id']}")
                else:
                    section_persona[sec] = None
            else:
                section_persona[sec] = None

    # Prepare data for each section with pre-assigned casino rotation lists and persona scenarios
    # Prepend relative time to evolution data so the AI knows when the old review was written
    relative_time = (evolution_facts or {}).get("_relative_time", "")
    def _get_evolution_for_section(sec):
        facts = (evolution_facts or {}).get(sec, "")
        if facts and facts.strip() and relative_time:
            return f"(The previous review was written {relative_time}.)\n{facts}"
        return facts

    section_data = [
        (sec, secs[sec], templates, sorted_comments, casino, btc_str, section_assignments.get(sec, []), _get_evolution_for_section(sec), section_persona.get(sec))
        for sec in section_order if sec in secs
    ]

    # Generate sections in parallel with max 3 workers to avoid API rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks and maintain order
        future_to_section = {
            executor.submit(generate_section_with_assignment, data): data[0]
            for data in section_data
        }

        # Collect results in the original order
        results = {}
        for future in concurrent.futures.as_completed(future_to_section):
            section_name = future_to_section[future]
            try:
                results[section_name] = future.result()
            except Exception as e:
                print(f"Error in parallel generation for {section_name}: {e}")
                results[section_name] = f"**{section_name}**\n[Error: {str(e)}]\n"

    # Return results in the original section order
    return [results[sec] for sec in section_order if sec in results]

def generate_section_with_assignment(section_data: Tuple) -> str:
    """Generate section with pre-assigned rotation list of casinos"""
    sec, content, templates, sorted_comments, casino, btc_str, casino_rotation_list, evolution_data, persona_scenario = section_data

    # Define section configurations
    section_configs = {
        "General": ("BaseGuidelinesClaude", "StructureTemplateGeneral", call_claude),
        "Payments": ("BaseGuidelinesClaude", "StructureTemplatePayments", call_claude),
        "Games": ("BaseGuidelinesClaude", "StructureTemplateGames", call_claude),
        "Responsible Gambling": ("BaseGuidelinesClaude", "StructureTemplateResponsible", call_claude),
        "FAQ": ("BaseGuidelinesClaude", "StructureTemplateFAQ", call_claude),
    }

    try:
        guidelines_file, structure_file, fn = section_configs[sec]

        # Get templates from cached data
        guidelines = templates.get(guidelines_file)
        structure = templates.get(structure_file)
        prompt_template = templates.get('PromptTemplate')

        if not guidelines or not structure or not prompt_template:
            return f"**{sec}**\n[Error: Missing templates for section {sec}]\n"

        # Get comments for this specific section
        section_comments = ""
        if sorted_comments.get(sec, "").strip():
            section_comments = f"\n\nCRITICAL USER FEEDBACK - MUST INCLUDE ALL DETAILS:\n{sorted_comments[sec]}\n\nIMPORTANT: The above user feedback contains specific, detailed information that MUST be included in your review. Do NOT summarize, simplify, or condense this information. Include ALL details, numbers, steps, mechanisms, and specifics exactly as provided.\n\nFor each piece of feedback:\n- If there is NO similar question/topic already in the review, create a NEW question and answer pair\n- If there IS already a similar question/topic in the review, APPEND the new information to that existing question and answer\n- This information is factual and verified - include it comprehensively in the review section."

        # Build round-robin instruction for the prompt
        round_robin_instruction = ""
        if casino_rotation_list:
            casinos_str = "', '".join(casino_rotation_list)
            round_robin_instruction = f"\n\nIMPORTANT - Casino Comparison Rotation:\nWhen making comparisons to other casinos in this section, rotate through these casinos from the Top/Similar data in THIS ORDER: '{casinos_str}'.\n\nFor your FIRST comparison, use '{casino_rotation_list[0]}'. For your SECOND comparison, use '{casino_rotation_list[1] if len(casino_rotation_list) > 1 else casino_rotation_list[0]}'. Continue rotating through this list for any additional comparisons. This ensures variety and prevents any single casino from being mentioned too frequently."
            print(f"Section {sec}: Rotation list = {casino_rotation_list}")

        # Inject personalization instruction into structure template
        structure_with_persona = structure
        if persona_scenario:
            structure_with_persona = structure + build_personalization_block(persona_scenario)

        # Inject evolution data directly into main data so the AI sees old vs new together
        evolution_addition = ""
        if evolution_data and evolution_data.strip():
            evolution_addition = f"\n\nPREVIOUS REVIEW DATA FOR COMPARISON (you MUST mention any differences between old and current data):\n{evolution_data}"

        prompt = prompt_template.format(
            casino=casino,
            section=sec,
            guidelines=guidelines,
            structure=structure_with_persona,
            main=content["main"] + section_comments + evolution_addition,
            top=content["top"],
            sim=content["sim"],
            btc_value=btc_str
        ) + round_robin_instruction

        review = fn(prompt)
        return f"**{sec}**\n{review}\n"

    except Exception as e:
        print(f"Error generating section {sec}: {e}")
        return f"**{sec}**\n[Error generating section: {str(e)}]\n"

def write_review_link_to_sheet(link):
    """Write the review link to cell B7 in the spreadsheet."""
    creds = get_service_account_credentials()
    sheets = build("sheets", "v4", credentials=creds)
    body = {"values": [[link]]}
    sheets.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID, 
        range=f"{SHEET_NAME}!B7", 
        valueInputOption="RAW", 
        body=body
    ).execute()

def insert_parsed_text_with_formatting(docs_service, doc_id, review_text):
    # Parse the text into clean text and extract formatting positions
    plain_text = ""
    formatting_requests = []
    cursor = 1  # Google Docs uses 1-based index after the title

    pattern = r'(\*\*(.*?)\*\*|\[([^\]]+?)\]\((https?://[^\)]+)\))'
    last_end = 0

    for match in re.finditer(pattern, review_text):
        start, end = match.span()
        before_text = review_text[last_end:start]
        plain_text += before_text
        cursor_start = cursor + len(before_text)

        if match.group(2):  # Bold (**text**)
            bold_text = match.group(2)
            plain_text += bold_text
            formatting_requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": cursor_start, "endIndex": cursor_start + len(bold_text)},
                    "textStyle": {"bold": True},
                    "fields": "bold"
                }
            })
            cursor += len(before_text) + len(bold_text)

        elif match.group(3) and match.group(4):  # Link [text](url)
            link_text = match.group(3)
            url = match.group(4)
            plain_text += link_text
            formatting_requests.append({
                "updateTextStyle": {
                    "range": {"startIndex": cursor_start, "endIndex": cursor_start + len(link_text)},
                    "textStyle": {"link": {"url": url}},
                    "fields": "link"
                }
            })
            cursor += len(before_text) + len(link_text)

        last_end = end

    remaining_text = review_text[last_end:]
    plain_text += remaining_text

    #  Insert clean plain text first
    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": [{"insertText": {"location": {"index": 1}, "text": plain_text}}]}
    ).execute()

    title_line = plain_text.split('\n', 1)[0]
    title_start = 1
    title_end = title_start + len(title_line)

    formatting_requests.insert(0, {
    "updateParagraphStyle": {
        "range": {"startIndex": title_start, "endIndex": title_end},
        "paragraphStyle": {"namedStyleType": "TITLE"},
        "fields": "namedStyleType"
        }
    })

    # Apply inline bold & links
    if formatting_requests:
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": formatting_requests}
        ).execute()

    doc = docs_service.documents().get(documentId=doc_id).execute()
    header_requests = []
    section_titles = ["Overview", "General", "Payments", "Games", "Responsible Gambling", "FAQ"]

    for element in doc.get('body', {}).get('content', []):
        if 'paragraph' in element:
            paragraph = element['paragraph']
            paragraph_text = ''.join(
                elem['textRun']['content']
                for elem in paragraph.get('elements', [])
                if 'textRun' in elem
            ).strip()

            # Check if this is a section title
            if paragraph_text in section_titles:
                # Find the exact start and end from element indexes
                start_index = element.get('startIndex')
                end_index = element.get('endIndex')
                if start_index is not None and end_index is not None:
                    header_requests.append({
                        "updateTextStyle": {
                            "range": {"startIndex": start_index, "endIndex": end_index - 1},  # exclude trailing newline
                            "textStyle": {"bold": True, "fontSize": {"magnitude": 16, "unit": "PT"}},
                            "fields": "bold,fontSize"
                        }
                    })

    # Apply section headers formatting
    if header_requests:
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": header_requests}
        ).execute()

def create_google_doc_in_folder(docs_service, drive_service, folder_id, doc_title, review_text):
    doc_id = docs_service.documents().create(body={"title": doc_title}).execute()["documentId"]
    insert_parsed_text_with_formatting(docs_service, doc_id, review_text)

    file = drive_service.files().get(fileId=doc_id, fields="parents").execute()
    previous_parents = ",".join(file.get('parents', []))
    drive_service.files().update(fileId=doc_id, addParents=folder_id, removeParents=previous_parents, fields="id, parents").execute()
    return doc_id

def find_existing_doc(drive_service, folder_id, title):
    query = f"name='{title}' and '{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

def main():
    st.set_page_config(page_title="Jacob Factual Review Generator", layout="centered", initial_sidebar_state="collapsed")

    # Initialize session state
    if 'review_completed' not in st.session_state:
        st.session_state.review_completed = False
        st.session_state.review_url = None
        st.session_state.casino_name = None


    # If review is already completed, show the success message
    if st.session_state.review_completed:
        st.success("Factual review successfully generated in Jacob structure!")
        if st.session_state.review_url:
            st.info(f"Review link: {st.session_state.review_url}")
        if st.session_state.get('askgamblers_debug'):
            st.caption(st.session_state.askgamblers_debug)
        if st.session_state.get('evolution_debug'):
            st.caption(st.session_state.evolution_debug)

        # Add a button to generate a new review
        if st.button("Write New Review", type="primary"):
            st.session_state.review_completed = False
            st.session_state.review_url = None
            st.session_state.casino_name = None
            st.rerun()
        return
    
    # Get casino name first to show in the interface
    try:
        user_creds = get_service_account_credentials()
        casino, _, _, _ = get_cached_casino_data()
        st.session_state.casino_name = casino
    except Exception as e:
        st.error(f"❌ Error loading casino data: {e}")
        return
    
    # Show casino name and generate button
    st.markdown(f"## Ready to write a factual review for: **{casino}**")
    st.markdown("The review will be generated in the Jacob structure with all factual data and comments.")
    
    # Only generate review when button is clicked
    if st.button("Write Review", type="primary", use_container_width=True):
        # Show progress message
        progress_placeholder = st.empty()
        progress_placeholder.markdown("## Writing review, please wait...")
        
        try:
            docs_service = build("docs", "v1", credentials=user_creds)
            drive_service = build("drive", "v3", credentials=user_creds)

            # Load all data in parallel
            progress_placeholder.markdown("## Loading templates and data...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all data loading tasks
                templates_future = executor.submit(get_all_templates)
                casino_data_future = executor.submit(get_cached_casino_data)
                btc_future = executor.submit(
                    lambda: requests.get(
                        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                        headers={"Accepts": "application/json", "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY},
                        params={"symbol": "BTC", "convert": "USD"}
                    ).json().get("data", {}).get("BTC", {}).get("quote", {}).get("USD", {}).get("price")
                )
                
                # Collect results
                templates = templates_future.result()
                casino, secs, comments, casino_id = casino_data_future.result()
                price = btc_future.result()
            
            btc_str = f"1 BTC = ${price:,.2f}" if price else "[BTC price unavailable]"
            
            # Check if all required templates were loaded
            required_templates = ['PromptTemplate', 'BaseGuidelinesClaude', 'BaseGuidelinesGrok']
            missing_templates = [t for t in required_templates if not templates.get(t)]
            if missing_templates:
                st.error(f"Error: Could not fetch required templates: {', '.join(missing_templates)}")
                return
            
            # Sort comments, fetch reputation, fetch old review, and scrape player feedback -- all in parallel
            progress_placeholder.markdown("## Sorting comments, fetching reputation, old review & player feedback...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                comments_future = executor.submit(sort_comments_by_section, comments)
                reputation_future = executor.submit(get_casino_reputation_summary, casino)
                evolution_future = executor.submit(fetch_and_extract_evolution_data, casino_id, casino)
                feedback_future = executor.submit(scrape_player_feedback, casino)
                sorted_comments = comments_future.result()
                reputation_summary = reputation_future.result()
                try:
                    evolution_facts, evolution_status = evolution_future.result()
                except Exception as e:
                    evolution_facts = {}
                    evolution_status = f"Old review fetch error: {e}"
                    print(f"Evolution data fetch failed: {e}")
                st.session_state.evolution_debug = evolution_status
                try:
                    feedback_data = feedback_future.result(timeout=90)
                except Exception as e:
                    feedback_data = {}
                    print(f"Player feedback scraper failed: {e}")

            if feedback_data and feedback_data.get("reviews"):
                review_count = feedback_data["total_count"]
                source = feedback_data["source_str"]
                print(f"Player feedback: {review_count} reviews from {source}")
                st.session_state.askgamblers_debug = f"Player feedback: {review_count} reviews from {source}"
            else:
                print("No player feedback returned")
                st.session_state.askgamblers_debug = "No player feedback returned"

            # Generate withdrawal summary early so it's available for FAQ section generation
            withdrawal_summary = ""
            if feedback_data and feedback_data.get("reviews"):
                withdrawal_summary = generate_withdrawal_player_summary(casino, feedback_data)

            # Inject reputation data into General section's main data
            if reputation_summary and "General" in secs:
                secs["General"]["main"] += f"\n\nREPUTATION DATA (from online sources):\n{reputation_summary}"

            # Inject Bonuses data into General and Payments (bonus questions now live in those sections)
            if "Bonuses" in secs:
                bonus_main = secs["Bonuses"]["main"]
                bonus_top = secs["Bonuses"]["top"]
                bonus_sim = secs["Bonuses"]["sim"]
                if "General" in secs:
                    secs["General"]["main"] += f"\n\nBONUS DATA:\n{bonus_main}"
                    if bonus_top:
                        secs["General"]["top"] += f"\n\nBonus comparison (Top Casinos):\n{bonus_top}"
                    if bonus_sim:
                        secs["General"]["sim"] += f"\n\nBonus comparison (Similar Casinos):\n{bonus_sim}"
                if "Payments" in secs:
                    secs["Payments"]["main"] += f"\n\nBONUS DATA (for bonus terms affecting withdrawals):\n{bonus_main}"
                del secs["Bonuses"]

            # Create FAQ section from General + Payments data (FAQ questions need VPN, anon, support, limits data)
            if "General" in secs:
                faq_main = secs["General"]["main"]
                if "Payments" in secs:
                    faq_main += f"\n\nPAYMENTS DATA (for highroller assessment):\n{secs['Payments']['main']}"
                if withdrawal_summary:
                    faq_main += f"\n\nPLAYER WITHDRAWAL FEEDBACK (from scraped reviews -- use for highroller withdrawal track record assessment):\n{withdrawal_summary}"
                secs["FAQ"] = {
                    "main": faq_main,
                    "top": secs["General"]["top"],
                    "sim": secs["General"]["sim"],
                }

            # Generate all sections in parallel
            progress_placeholder.markdown("## Generating review sections in parallel...")

            parallel_results = generate_sections_parallel(casino, secs, sorted_comments, templates, btc_str, evolution_facts)

            # Insert dynamically generated content from scraped reviews into section outputs
            if feedback_data and feedback_data.get("reviews"):
                # General section: insert "What do players say" after legit question
                general_summary = generate_general_player_summary(casino, feedback_data)
                if general_summary and len(parallel_results) > 0:
                    general_text = parallel_results[0]
                    first_q_pos = general_text.find('## Q:')
                    if first_q_pos >= 0:
                        second_q_match = re.search(r'\n(## Q:)', general_text[first_q_pos + 5:])
                        if second_q_match:
                            insert_pos = first_q_pos + 5 + second_q_match.start()
                            parallel_results[0] = general_text[:insert_pos] + general_summary + "\n" + general_text[insert_pos:]
                        else:
                            parallel_results[0] = general_text.rstrip('\n') + "\n" + general_summary + "\n"
                    else:
                        parallel_results[0] = general_text.rstrip('\n') + "\n" + general_summary + "\n"

                # Payments section: append withdrawal complaints Q&A
                if withdrawal_summary and len(parallel_results) > 1:
                    withdrawal_qa = f"\n## Q: Do players have any complaints regarding withdrawals?\n\n{withdrawal_summary}"
                    parallel_results[1] = parallel_results[1].rstrip('\n') + "\n" + withdrawal_qa + "\n"

            # Combine results into final factual review
            factual_review = "\n".join([f"{casino} review\n"] + parallel_results)

            # Upload to Google Docs
            progress_placeholder.markdown("## Uploading to Google Drive...")
            doc_title = f"{casino} Review - Jacob Factual"
            existing_doc_id = find_existing_doc(drive_service, FOLDER_ID, doc_title)

            if existing_doc_id:
                drive_service.files().delete(fileId=existing_doc_id).execute()

            doc_id = create_google_doc_in_folder(docs_service, drive_service, FOLDER_ID, doc_title, factual_review)
            doc_url = f"https://docs.google.com/document/d/{doc_id}"

            # Write the review link to the spreadsheet
            write_review_link_to_sheet(doc_url)

            # Mark as completed
            st.session_state.review_completed = True
            st.session_state.review_url = doc_url
            st.session_state.casino_name = casino

            # Clear progress message
            progress_placeholder.empty()
            st.success(f"Factual review generated successfully!")
            st.info(f"Review link: {doc_url}")
            st.rerun()

        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()