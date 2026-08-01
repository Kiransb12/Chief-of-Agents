"""
Tool definitions for Phase 2.
Includes:
- Web Search (DuckDuckGo search scraper)
- Calendar management (SQLite)
- Weather (Open-Meteo free weather API)
"""
import os
import re
import time
import sqlite3
import threading
import urllib.parse
import webbrowser
import requests
import pyautogui

CALENDAR_DB = "./data/calendar.db"
_CALENDAR_LOCK = threading.Lock()


def _get_calendar_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CALENDAR_DB), exist_ok=True)
    conn = sqlite3.connect(CALENDAR_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            date_time TEXT NOT NULL,
            duration_minutes INTEGER NOT NULL DEFAULT 30,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def get_calendar_events() -> str:
    """Retrieve all scheduled events from the personal calendar."""
    with _CALENDAR_LOCK, _get_calendar_conn() as conn:
        rows = conn.execute(
            "SELECT title, date_time, duration_minutes FROM events ORDER BY id"
        ).fetchall()
    if not rows:
        return "Your calendar is empty."
    res = ["Your Calendar Events:"]
    for idx, (title, date_time, duration_minutes) in enumerate(rows):
        res.append(f"{idx+1}. '{title}' on {date_time} ({duration_minutes} mins)")
    return "\n".join(res)


def create_calendar_event(title: str, date_time: str, duration_minutes: int = 30) -> str:
    """Schedule a new event on the calendar.

    Args:
        title: The title or description of the event.
        date_time: The date and time of the event (e.g. 'tomorrow at 10 AM', '2026-07-17 14:00').
        duration_minutes: The duration of the event in minutes. Defaults to 30.
    """
    with _CALENDAR_LOCK, _get_calendar_conn() as conn:
        conn.execute(
            "INSERT INTO events (title, date_time, duration_minutes) VALUES (?, ?, ?)",
            (title, date_time, duration_minutes),
        )
    return f"Successfully scheduled event '{title}' for {date_time} ({duration_minutes} minutes)."


def search_web(query: str) -> str:
    """Search the web for the given query and return key snippets.

    Args:
        query: The search query.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        # Try DuckDuckGo Instant Answer API first
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
        resp = requests.get(url, headers=headers, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("AbstractText", "")
            if abstract:
                return f"[DuckDuckGo Abstract] {abstract}"

        # Fallback to DuckDuckGo Lite search POST
        url = "https://lite.duckduckgo.com/lite/"
        data = {"q": query}
        resp = requests.post(url, headers=headers, data=data, timeout=3)
        if resp.status_code == 200:
            html = resp.text
            snippets = []
            matches = re.findall(
                r"<td[^>]*class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
                html,
                re.DOTALL,
            )
            for m in matches[:3]:
                clean = re.sub(r"<[^>]+>", "", m).strip()
                snippets.append(clean)
            if snippets:
                return "\n".join(f"- {s}" for s in snippets)
        return "No search results found."
    except Exception as e:
        return f"Search error: {e}"


def get_live_weather(location: str) -> str:
    """Get the current weather conditions for a given location.

    Args:
        location: City or location name (e.g., 'Bangalore', 'New York').
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        # 1. Geocode location to lat/lon
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, headers=headers, timeout=3)
        if geo_resp.status_code != 200:
            return f"Failed to geocode location '{location}'"

        geo_data = geo_resp.json()
        results = geo_data.get("results", [])
        if not results:
            return f"Could not find location '{location}'."

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
        name = results[0]["name"]
        country = results[0].get("country", "")

        # 2. Get current weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        w_resp = requests.get(weather_url, headers=headers, timeout=3)
        if w_resp.status_code != 200:
            return f"Failed to retrieve weather for '{location}'"

        w_data = w_resp.json()
        current = w_data.get("current_weather", {})
        if not current:
            return f"Weather data not available for '{location}'."

        temp = current.get("temperature", "unknown")
        windspeed = current.get("windspeed", "unknown")
        weathercode = current.get("weathercode", 0)

        conditions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow fall",
            73: "Moderate snow fall",
            75: "Heavy snow fall",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
        }
        cond = conditions.get(weathercode, f"Code {weathercode}")

        return f"Current Weather in {name}, {country}: {temp}°C, {cond}. Wind Speed: {windspeed} km/h."
    except Exception as e:
        return f"Weather error: {e}"


def open_browser_and_search(query: str) -> str:
    """Open the default web browser and search for a query or open a URL.

    Args:
        query: The search query or URL to open.
    """
    try:
        # Check if the query is a URL
        stripped_query = query.strip()
        if stripped_query.startswith(("http://", "https://")):
            url = stripped_query
        elif re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/.*)?$", stripped_query):
            url = f"https://{stripped_query}"
        else:
            encoded_query = urllib.parse.quote(stripped_query)
            url = f"https://www.google.com/search?q={encoded_query}"
        
        webbrowser.open(url)
        return f"Successfully opened default browser to: {url}"
    except Exception as e:
        return f"Failed to open browser and search: {e}"


def scroll_webpage(direction: str, amount: int = 5) -> str:
    """Scroll the active webpage window up or down.

    Args:
        direction: The direction to scroll ('up' or 'down').
        amount: The number of scroll increments (default 5).
    """
    try:
        import pyautogui
        import time
        time.sleep(0.5)
        # Positive scrolls up, negative scrolls down.
        clicks = amount * 120
        if direction.lower() == "down":
            clicks = -clicks
        
        pyautogui.scroll(clicks)
        return f"Successfully scrolled {direction} by {amount} units."
    except Exception as e:
        return f"Failed to scroll webpage: {e}"


def search_in_page(query: str) -> str:
    """Perform a text search (Ctrl+F) on the active webpage for the given query.

    Args:
        query: The text to search for on the page.
    """
    try:
        import pyautogui
        import time
        time.sleep(0.5)
        pyautogui.hotkey('ctrl', 'f')
        time.sleep(0.2)
        pyautogui.write(query)
        time.sleep(0.1)
        pyautogui.press('enter')
        return f"Successfully triggered search for '{query}' on active page."
    except Exception as e:
        return f"Failed to search in page: {e}"


def whatsapp_send_message(phone: str, message: str) -> str:
    """Open WhatsApp Web to send a message to a phone number or a contact name.

    Args:
        phone: The phone number or contact name.
        message: The message content to send.
    """
    try:
        # Check if the input phone is a phone number or a contact name
        # A phone number can only contain digits, spaces, plus, hyphens, and parentheses
        is_phone = bool(re.match(r"^\+?[\d\s\-()]+$", phone.strip()))

        if is_phone:
            clean_phone = re.sub(r"[^\d+]", "", phone)
            
            def run_send():
                try:
                    encoded_message = urllib.parse.quote(message)
                    url = f"https://web.whatsapp.com/send?phone={clean_phone}&text={encoded_message}"
                    webbrowser.open(url)
                    # Wait 15 seconds for page load and synchronization
                    time.sleep(15)
                    # Press enter to send
                    pyautogui.press('enter')
                except Exception:
                    pass

            # Run in background so we don't block the orchestrator loop
            thread = threading.Thread(target=run_send, daemon=True)
            thread.start()
            
            return f"Opening WhatsApp Web to send message to phone: {clean_phone}."
        else:
            # It's a contact name!
            contact_name = phone.strip()

            def run_contact_send():
                try:
                    # Open WhatsApp Web main page
                    webbrowser.open("https://web.whatsapp.com")
                    # Wait for WhatsApp Web to load
                    time.sleep(15)
                    # Focus search box (shortcut: Ctrl+Alt+/)
                    pyautogui.hotkey('ctrl', 'alt', '/')
                    time.sleep(0.5)
                    # Type contact name
                    pyautogui.write(contact_name)
                    time.sleep(1.5) # Wait for search results
                    # Press Enter to open the chat
                    pyautogui.press('enter')
                    time.sleep(1.0)
                    # Type the message
                    pyautogui.write(message)
                    time.sleep(0.5)
                    # Press Enter to send
                    pyautogui.press('enter')
                except Exception:
                    pass

            thread = threading.Thread(target=run_contact_send, daemon=True)
            thread.start()

            return f"Opening WhatsApp Web to search and send message to contact: '{contact_name}'."
    except Exception as e:
        return f"Failed to send WhatsApp message: {e}"


def go_to_main_page() -> str:
    """Open the default web browser and navigate to the agent's main dashboard page."""
    try:
        url = "http://127.0.0.1:8000/"
        webbrowser.open(url)
        return f"Successfully opened main agent page at: {url}"
    except Exception as e:
        return f"Failed to navigate to main page: {e}"


def open_new_tab(url: str = "https://www.google.com") -> str:
    """Open a new tab in the default web browser with the given URL.

    Args:
        url: The URL to open in the new tab (default is Google).
    """
    try:
        import webbrowser
        webbrowser.open_new_tab(url)
        return f"Successfully opened a new tab with: {url}"
    except Exception as e:
        return f"Failed to open new tab: {e}"



