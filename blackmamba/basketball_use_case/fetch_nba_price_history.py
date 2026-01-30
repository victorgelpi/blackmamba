# fetch_nba_price_history.py

from pathlib import Path
from datetime import datetime
import json

import pandas as pd

from polymarket.data_collection import DataCollection


# ----------------- NBA TEAM HELPERS -----------------

NBA_TEAMS = {
    # East
    "hawks", "celtics", "nets", "hornets", "bulls",
    "cavaliers", "cavs", "pistons", "pacers", "heat",
    "bucks", "knicks", "magic", "76ers", "sixers",
    "raptors", "wizards",
    # West
    "mavericks", "mavs", "nuggets", "warriors", "gsw",
    "rockets", "lakers", "clippers", "grizzlies", "grizz",
    "timberwolves", "wolves", "pelicans", "pels",
    "thunder", "suns", "blazers", "trail blazers",
    "kings", "spurs", "jazz",
}

TEAM_NORMALIZATION = {
    "portland trail blazers": "blazers",
    "san antonio spurs": "spurs",
    "los angeles lakers": "lakers",
    "los angeles clippers": "clippers",
    "golden state warriors": "warriors",
    "oklahoma city thunder": "thunder",
    "new orleans pelicans": "pelicans",
    "minnesota timberwolves": "timberwolves",
}


def is_nba_event(event: dict) -> bool:
    """
    Event is NBA if:
    1. Title/question contains 'vs.' (typical game market)
    2. Left and right teams are both real NBA teams
    """
    title = (event.get("title") or event.get("question") or "").lower()
    if "vs." not in title:
        return False

    try:
        left, right = title.split("vs.")
    except ValueError:
        return False

    left = left.strip()
    right = right.strip()

    # Normalize multi-word team names
    for full, short in TEAM_NORMALIZATION.items():
        if full in left:
            left = short
        if full in right:
            right = short

    # Keep only last word ("trail blazers" -> "blazers")
    left_token = left.split()[-1]
    right_token = right.split()[-1]

    return (left_token in NBA_TEAMS) and (right_token in NBA_TEAMS)


# ----------------- HELPERS FOR PRICE HISTORY -----------------


def parse_closed_ts(ev: dict, market: dict) -> int | None:
    # 1. Try market 'closedTime' (Standard for closed markets)
    closed_raw = market.get("closedTime")
    if closed_raw:
        try:
            # Handles "2024-10-04T20:38:17Z" and "2024-10-04 20:38:17+00"
            clean_raw = closed_raw.replace("Z", "+00:00").replace(" ", "T")
            dt = datetime.fromisoformat(clean_raw)
            return int(dt.timestamp())
        except ValueError:
            pass

    # 2. Try event 'end date' (DataCollection usually parses this to datetime)
    end_dt = DataCollection.get_field(ev, "end date")
    if isinstance(end_dt, datetime):
        return int(end_dt.timestamp())
        
    return None

def fetch_history_for_token(token_id: str, start_ts: int, end_ts: int):
    """
    Fetch price history using Mode 2 (Specific Date Range).
    This guarantees we look at the time the market was actually alive.
    """
    # Calculate a safe window
    # If the market duration is short, use higher fidelity (1m or 5m)
    # If long, use hourly (60m)
    duration_hours = (end_ts - start_ts) / 3600
    
    # Configuration strategy
    if duration_hours < 48:
        fidelity = 5   # 5-minute bars for short windows (games)
    else:
        fidelity = 60  # Hourly bars for longer windows
        
    try:
        # Mode 2: start_ts + end_ts + fidelity
        prices = DataCollection.price_history(
            market=token_id,
            start_ts=start_ts,
            end_ts=end_ts,
            fidelity=fidelity
        )
        return prices
    except Exception as e:
        print(f"[WARN] price_history error for token {token_id}: {e}")
        return []
# ----------------- MAIN SCRIPT -----------------


def main():
    BASE_DIR = Path(__file__).resolve().parent
    OUT_DIR = BASE_DIR / "data" / "processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start_date = datetime(2024, 10, 1)
    end_date = datetime(2025, 6, 30, 23, 59, 59)

    print(f"Fetching closed events from {start_date} to {end_date}...")

    events = DataCollection.closed_events(
        start_date_min=start_date,
        end_date_max=end_date,
        limit=5000,
        force_large=True,
    )

    print(f"Total closed events fetched in window: {len(events)}")

    nba_events = [ev for ev in events if is_nba_event(ev)]
    print(f"Detected {len(nba_events)} NBA/basketball events in this window.")

    if not nba_events:
        print("No NBA events found with current heuristics + date range.")
        return

    event_rows = []
    price_rows = []

    for ev in nba_events:
        fields = DataCollection.extract_fields(
            ev,
            ["id", "title", "question", "start date", "end date", "volume", "tags"],
        )

        event_id = fields.get("id")
        title = fields.get("title") or fields.get("question")

        markets = ev.get("markets") or []
        if not markets:
            continue

        market = markets[0]

        # Use helpers for outcomes/prices/token IDs
        outcomes = DataCollection.get_field(market, "outcomes")
        outcome_prices = DataCollection.get_field(market, "outcomePrices")
        token_ids = DataCollection.getClobTokenId(market)

        last_trade = market.get("lastTradePrice")
        uma_status = market.get("umaResolutionStatus")

        event_rows.append(
            {
                "event_id": event_id,
                "title": title,
                "question": fields.get("question"),
                "start_date": fields.get("start date"),
                "end_date": fields.get("end date"),
                "volume": fields.get("volume"),
                "tags": fields.get("tags"),
                "outcomes": outcomes,
                "outcome_prices": outcome_prices,
                "uma_resolution_status": uma_status,
                "last_trade_price": last_trade,
                "token_ids": token_ids,
            }
        )

        if not token_ids:
            print(f"[INFO] No token IDs for event {event_id} ({title})")
            continue

        yes_token = str(token_ids[0])  # left-side team


        # --- UPDATED LOGIC START ---
        
        # 1. Determine End Timestamp
        # Try market closedTime first, then event end date
        end_ts = parse_closed_ts(ev, market)
        if not end_ts:
            # Fallback: if we can't find a close time, skip to avoid "now()" errors
            print(f"[WARN] Could not determine close time for {event_id}. Skipping.")
            continue

        # 2. Determine Start Timestamp
        # Use the event's start date (which you already extracted)
        start_dt = fields.get("start date")
        if not start_dt:
            print(f"[WARN] No start date for {event_id}. Skipping.")
            continue
        
        # Add a buffer: Start fetching 24 hours before the game starts 
        # to capture pre-game betting volume.
        start_ts = int(start_dt.timestamp()) - (24 * 3600)
        
        # Ensure start is before end
        if start_ts >= end_ts:
            start_ts = end_ts - (24 * 3600) # Force 24h window if dates are weird

        # 3. Fetch with explicit window
        prices = fetch_history_for_token(yes_token, start_ts=start_ts, end_ts=end_ts)
        
        # --- UPDATED LOGIC END ---

        if not prices:
            print(f"[INFO] Empty history for event {event_id}, token {yes_token}")
            continue

        if not prices:
            print(f"[INFO] Empty history for event {event_id}, token {yes_token}")
            continue

        for p in prices:
            t = p.get("t")
            val = p.get("p")
            if t is None or val is None:
                continue
            price_rows.append(
                {
                    "event_id": event_id,
                    "title": title,
                    "token_id": yes_token,
                    "timestamp": t,
                    "price": float(val),
                }
            )

    events_df = pd.DataFrame(event_rows)
    prices_df = pd.DataFrame(price_rows)

    events_out = OUT_DIR / "nba_events_2024_2025_with_markets.csv"
    prices_out = OUT_DIR / "nba_event_price_history.csv"

    events_df.to_csv(events_out, index=False)
    prices_df.to_csv(prices_out, index=False)

    print(f"\nSaved {len(events_df)} NBA events to: {events_out}")
    print(f"Saved {len(prices_df)} price points to: {prices_out}\n")

    if not events_df.empty:
        print(events_df[["event_id", "title", "outcomes", "outcome_prices"]].head())
    if not prices_df.empty:
        print()
        print(prices_df.head())


if __name__ == "__main__":
    main()
