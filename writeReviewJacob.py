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
        'StructureTemplateBonuses'
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(get_file_content_from_github, filename): filename 
                         for filename in files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                templates[filename] = future.result()
            except Exception as e:
                print(f"Error fetching template {filename}: {e}")
                templates[filename] = None
    
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
    
    return casino, data, all_comments

def get_cached_casino_data():
    """Get casino data without caching to prevent tone interference"""
    return get_selected_casino_data()

# AI CLIENTS
client = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

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
    return anthropic.messages.create(model="claude-sonnet-4-20250514", max_tokens=1200, temperature=0.45, messages=[{"role": "user", "content": full_prompt}]).content[0].text.strip()

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
        return {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "Bonuses": ""}

    prompt = f"""Please analyze the following feedback comments and sort them by the casino review sections they belong to.

Comments:
{comments}

Sections:
- General (overall casino experience, VPN support, reputation, establishment date, etc.)
- Payments (deposits, withdrawals, KYC, payment methods, processing times, etc.)
- Games (game selection, slots, table games, live casino, game providers, etc.)
- Responsible Gambling (limits, self-exclusion, problem gambling tools, etc.)
- Bonuses (welcome bonus, promotions, bonus terms, wagering requirements, etc.)

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

**Bonuses**
[relevant comments here or leave empty]"""
    
    try:
        response = call_claude(prompt)
        # Parse the response into a dictionary
        sections = {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "Bonuses": ""}
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

        # Redirect Bonuses comments to General section (temporary until Bonuses structure template is ready)
        if sections.get("Bonuses", "").strip():
            print("Redirecting Bonuses comments to General section")
            if sections["General"]:
                sections["General"] += " " + sections["Bonuses"]
            else:
                sections["General"] = sections["Bonuses"]
            sections["Bonuses"] = ""

        return sections
    except Exception as e:
        print(f"Error sorting comments: {e}")
        # Fallback: return empty sections
        return {"General": "", "Payments": "", "Games": "", "Responsible Gambling": "", "Bonuses": ""}

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


def _prepare_reviews_for_prompt(reviews, max_reviews=20):
    """Prepare a compact text block of reviews for an AI prompt."""
    lines = []
    for i, rev in enumerate(reviews[:max_reviews], 1):
        rating = rev.get('rating', 'N/A')
        text = (rev.get('text') or rev.get('title') or '').strip()
        if not text:
            continue
        # Truncate long reviews
        if len(text) > 300:
            text = text[:300].rsplit(' ', 1)[0] + "..."
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

    prompt = f"""You are writing a section for a casino review article about {casino_name}.
Summarize what players say about {casino_name} based on {total} recent reviews from {source_str}.

RULES:
- Write 2-3 short paragraphs as a Q&A answer (the question is "What do players say about {casino_name}?")
- Start with something like "Based on {total} recent player reviews from {source_str}, ..."
- Focus primarily on what players praise (good support, fast payouts, game variety, etc.)
- Lightly mention common complaints if any exist, but don't dwell on them
- Keep in mind many negative reviews come from emotional post-loss players, so weigh positives more heavily
- Do NOT mention that players might be biased or salty -- just focus on the actual feedback
- Do NOT use em dashes (use commas or periods instead)
- Use first person "I" and address the reader as "you"
- Bold key highlights with **double asterisks**
- Keep it concise, 80-150 words total
- Do NOT include the question heading, just the answer text

PLAYER REVIEWS:
{reviews_text}"""

    try:
        summary = call_claude(prompt)
        result = f"\n## Q: What do players say about {casino_name}?\n\n{summary}"
        return result
    except Exception as e:
        print(f"General player summary AI call failed: {e}")
        return ""


def generate_payments_player_summary(casino_name, feedback_data):
    """Use Claude to summarize payment-related player reviews for the Payments section."""
    reviews = feedback_data["reviews"]
    source_str = feedback_data["source_str"]
    total = feedback_data["total_count"]

    # Filter to payment-related reviews
    payment_kw = ['withdrawal', 'withdraw', 'payout', 'cashout', 'deposit',
                   'payment', 'pending', 'processing', 'kyc', 'verification',
                   'cash out', 'pay out', 'waiting']
    payment_reviews = []
    for rev in reviews:
        content = (rev.get('text', '') + ' ' + rev.get('title', '')).lower()
        if any(kw in content for kw in payment_kw):
            payment_reviews.append(rev)

    if not payment_reviews:
        return ""

    reviews_text = _prepare_reviews_for_prompt(payment_reviews)
    if not reviews_text:
        return ""

    prompt = f"""You are writing a section for a casino review article about {casino_name}.
Summarize what players say about payments and withdrawals at {casino_name} based on player reviews from {source_str}.

RULES:
- Write 1-2 short paragraphs as a Q&A answer (the question is "What do players say about {casino_name} withdrawals?")
- Start with something like "Based on player feedback from {source_str}, ..."
- Focus primarily on positive payment experiences (fast payouts, smooth processing, etc.)
- Lightly mention withdrawal complaints if they exist, but don't overemphasize them
- Keep in mind many negative reviews come from emotional post-loss players, so weigh positives more heavily
- Do NOT mention that players might be biased or salty -- just focus on the actual feedback
- Do NOT use em dashes (use commas or periods instead)
- Use first person "I" and address the reader as "you"
- Bold key highlights with **double asterisks**
- Keep it concise, 60-120 words total
- Do NOT include the question heading, just the answer text

PAYMENT-RELATED PLAYER REVIEWS ({len(payment_reviews)} of {total} total reviews):
{reviews_text}"""

    try:
        summary = call_claude(prompt)
        result = f"\n## Q: What do players say about {casino_name} withdrawals?\n\n{summary}"
        return result
    except Exception as e:
        print(f"Payments player summary AI call failed: {e}")
        return ""


def generate_sections_parallel(casino: str, secs: Dict, sorted_comments: Dict, templates: Dict, btc_str: str) -> list:
    """Generate all sections in parallel while maintaining round-robin casino selection"""

    # Initialize tracker for used comparison casinos
    used_casinos_tracker = []

    # Pre-assign a rotation list of casinos to each section
    # We need to do this sequentially before parallel generation
    section_order = ["General", "Payments", "Games", "Responsible Gambling", "Bonuses"]
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

    # Prepare data for each section with pre-assigned casino rotation lists
    section_data = [
        (sec, secs[sec], templates, sorted_comments, casino, btc_str, section_assignments.get(sec, []))
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
    sec, content, templates, sorted_comments, casino, btc_str, casino_rotation_list = section_data

    # Define section configurations
    section_configs = {
        "General": ("BaseGuidelinesClaude", "StructureTemplateGeneral", call_claude),
        "Payments": ("BaseGuidelinesClaude", "StructureTemplatePayments", call_claude),
        "Games": ("BaseGuidelinesClaude", "StructureTemplateGames", call_claude),
        "Responsible Gambling": ("BaseGuidelinesGrok", "StructureTemplateResponsible", call_grok),
        "Bonuses": ("BaseGuidelinesClaude", "StructureTemplateBonuses", call_claude),
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

        prompt = prompt_template.format(
            casino=casino,
            section=sec,
            guidelines=guidelines,
            structure=structure,
            main=content["main"] + section_comments,
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
    section_titles = ["Overview", "General", "Payments", "Games", "Responsible Gambling", "Bonuses"]

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
        casino, _, _ = get_cached_casino_data()
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
                casino, secs, comments = casino_data_future.result()
                price = btc_future.result()
            
            btc_str = f"1 BTC = ${price:,.2f}" if price else "[BTC price unavailable]"
            
            # Check if all required templates were loaded
            required_templates = ['PromptTemplate', 'BaseGuidelinesClaude', 'BaseGuidelinesGrok']
            missing_templates = [t for t in required_templates if not templates.get(t)]
            if missing_templates:
                st.error(f"Error: Could not fetch required templates: {', '.join(missing_templates)}")
                return
            
            # Sort comments by section using AI
            progress_placeholder.markdown("## Sorting comments by section...")
            sorted_comments = sort_comments_by_section(comments)

            # Generate all sections in parallel + scrape AskGamblers in background
            progress_placeholder.markdown("## Generating review sections in parallel...")

            bg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            askgamblers_future = bg_executor.submit(scrape_player_feedback, casino)

            parallel_results = generate_sections_parallel(casino, secs, sorted_comments, templates, btc_str)

            # Collect scraper result (should be done by now since sections take longer)
            try:
                feedback_data = askgamblers_future.result(timeout=90)
            except Exception as e:
                feedback_data = {}
                print(f"Player feedback scraper failed: {e}")
            finally:
                bg_executor.shutdown(wait=False)

            # Generate AI summaries from scraped reviews
            if feedback_data and feedback_data.get("reviews"):
                review_count = feedback_data["total_count"]
                source = feedback_data["source_str"]
                print(f"Player feedback: {review_count} reviews from {source}")
                st.session_state.askgamblers_debug = f"Player feedback: {review_count} reviews from {source}"

                # AI call for General section
                general_summary = generate_general_player_summary(casino, feedback_data)
                if general_summary and len(parallel_results) > 0:
                    parallel_results[0] = parallel_results[0].rstrip('\n') + "\n" + general_summary + "\n"

                # AI call for Payments section
                payments_summary = generate_payments_player_summary(casino, feedback_data)
                if payments_summary and len(parallel_results) > 1:
                    parallel_results[1] = parallel_results[1].rstrip('\n') + "\n" + payments_summary + "\n"
            else:
                print("No player feedback returned")
                st.session_state.askgamblers_debug = "No player feedback returned"

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