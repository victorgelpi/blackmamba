from pathlib import Path
from datetime import datetime
import pandas as pd

from polymarket.data_collection import DataCollection


# ---------- 1. Simple classifier: is this event "NBA game"? ----------
# ---------- NBA team list ----------

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
    "kings", "spurs", "jazz"
}

# Normalize full team names → short names
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

    # Example structure: "trail blazers vs. spurs"
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

    # Keep only last word in multi-word names ("trail blazers" → "blazers")
    left_token = left.split()[-1]
    right_token = right.split()[-1]

    # Now check membership
    return (left_token in NBA_TEAMS) and (right_token in NBA_TEAMS)

# ---------- 2. Fetch events and find NBA games ----------
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

    rows = []
    for ev in nba_events:
        # top-level fields via extract_fields
        fields = DataCollection.extract_fields(
            ev,
            ["id", "title", "question", "start date", "end date", "volume", "tags"],
        )

        # --- pull market-level info (assume 1 main market per event) ---
        markets = ev.get("markets") or []
        if markets:
            m = markets[0]
            outcomes_raw = m.get("outcomes")
            outcome_prices_raw = m.get("outcomePrices")
            clob_token_ids_raw = m.get("clobTokenIds")

            try:
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            except Exception:
                outcomes = None

            try:
                outcome_prices = json.loads(outcome_prices_raw) if isinstance(outcome_prices_raw, str) else outcome_prices_raw
            except Exception:
                outcome_prices = None

            try:
                clob_token_ids = json.loads(clob_token_ids_raw) if isinstance(clob_token_ids_raw, str) else clob_token_ids_raw
            except Exception:
                clob_token_ids = None

            uma_status = m.get("umaResolutionStatus")
            last_trade = m.get("lastTradePrice")
            best_bid = m.get("bestBid")
            best_ask = m.get("bestAsk")
            condition_id = m.get("conditionId")
        else:
            outcomes = outcome_prices = clob_token_ids = None
            uma_status = last_trade = best_bid = best_ask = condition_id = None

        rows.append(
            {
                "event_id": fields.get("id"),
                "title": fields.get("title") or fields.get("question"),
                "question": fields.get("question"),
                "start_date": fields.get("start date"),
                "end_date": fields.get("end date"),
                "volume": fields.get("volume"),
                "tags": fields.get("tags"),

                # market-level / resolution-ish stuff
                "condition_id": condition_id,
                "outcomes": outcomes,
                "outcome_prices": outcome_prices,  # final payouts (1/0 after resolve)
                "uma_resolution_status": uma_status,
                "last_trade_price": last_trade,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "clob_token_ids": clob_token_ids,
            }
        )

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "nba_events_2024_2025_with_markets.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} NBA events with market info to: {out_path}\n")
    print(df[["event_id", "title", "start_date", "end_date", "outcomes", "outcome_prices", "uma_resolution_status"]].head())


if __name__ == "__main__":
    main()
