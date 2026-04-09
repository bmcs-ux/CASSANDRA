import requests
import json
import os
from datetime import datetime, timedelta, timezone

class NewsManager:
    def __init__(self, data_dir, logger):
        self.data_path = os.path.join(data_dir, "daily_news.json")
        self._log = logger
        self.high_impact_events = []
        self.last_sync_date = None

    def sync_news(self):
        """Mengambil berita dari API dan menyimpan ke JSON lokal."""
        try:
            self._log("[NEWS] Syncing daily economic calendar...")
            # Menggunakan API aggregator yang mengambil data ForexFactory
            # URL ini adalah contoh, pastikan menggunakan provider yang stabil
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            response = requests.get(url, timeout=15)

            if response.status_code == 200:
                all_events = response.json()
                # Filter hanya High Impact dan sesuaikan waktu
                self.high_impact_events = [
                    e for e in all_events if e.get('impact') == 'High'
                ]

                with open(self.data_path, 'w') as f:
                    json.dump(self.high_impact_events, f)

                self.last_sync_date = datetime.now().date()
                self._log(f"[NEWS] Sync complete. Found {len(self.high_impact_events)} High Impact events.")
                return True
        except Exception as e:
            self._log(f"[ERROR] News sync failed: {e}")
            return False

    def load_local_news(self):
        """Memuat data berita dari file lokal ke memori."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                self.high_impact_events = json.load(f)
            self._log(f"[NEWS] Loaded {len(self.high_impact_events)} events from local cache.")

    def is_currently_restricted(self):
        """Cek apakah sekarang masuk jendela dilarang (5 menit sebelum/sesudah)."""
        return self.get_news_status().get("is_restricted", False)

    def get_news_status(self):
        """Ringkas status gate news + hitung mundur event High Impact terdekat."""
        if not self.high_impact_events:
            return {
                "is_restricted": False,
                "active_event": None,
                "next_event": None,
                "seconds_to_next_event": None,
                "window_minutes": 5,
            }

        now_utc = datetime.utcnow()
        active_event = None
        next_event = None
        smallest_delta = None

        for event in self.high_impact_events:
            try:
                event_time = self._parse_event_time(event["date"])
            except (KeyError, ValueError, TypeError) as exc:
                self._log(f"[WARN] Skipping malformed news event: {exc}")
                continue

            start_window = event_time - timedelta(minutes=5)
            end_window = event_time + timedelta(minutes=5)

            if start_window <= now_utc <= end_window and active_event is None:
                active_event = {
                    "title": event.get("title"),
                    "country": event.get("country"),
                    "event_time_utc": event_time.isoformat(),
                    "window_start_utc": start_window.isoformat(),
                    "window_end_utc": end_window.isoformat(),
                }

            if event_time > now_utc:
                delta_seconds = int((event_time - now_utc).total_seconds())
                if smallest_delta is None or delta_seconds < smallest_delta:
                    smallest_delta = delta_seconds
                    next_event = {
                        "title": event.get("title"),
                        "country": event.get("country"),
                        "event_time_utc": event_time.isoformat(),
                    }

        if active_event:
            self._log(f"[BLOCK] Trade prohibited! News: {active_event['title']} ({active_event['country']})")

        return {
            "is_restricted": active_event is not None,
            "active_event": active_event,
            "next_event": next_event,
            "seconds_to_next_event": smallest_delta,
            "window_minutes": 5,
        }

    def _parse_event_time(self, date_str):
        # Helper untuk konversi format ISO ke UTC datetime tanpa dependensi eksternal.
        parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
