import csv
import json
import os
import logging
from datetime import datetime
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence

import config

logger = logging.getLogger(__name__)

# --- GENERIC JSON HELPERS ---

def save_json(data: Any, filename: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filename: str) -> Any:
    """Load data from JSON file (returns [] if not found)."""
    if not os.path.exists(filename):
        logger.warning(f"JSON file {filename} not found, returning empty list.")
        return []
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


# --- TOTO FUNCTIONS ---

def append_toto_csv(record: Mapping[str, Any], filename: str = None) -> None:
    """Append record to TOTO CSV file."""
    if filename is None:
        filename = str(config.TOTO_CSV_FILE)
    
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Bonus"])
        
        # Determine format
        dt_str = record["date"]
        if isinstance(dt_str, datetime):
            dt_str = dt_str.strftime("%Y-%m-%d")
        
        row = [dt_str, *record["numbers"], record.get("bonus", "")]
        writer.writerow(row)


def load_toto_history() -> List[MutableMapping[str, Any]]:
    """Load raw Toto history from CSV.
    
    CSV Format: Draw, Date, Winning Number 1-6, Additional Number, ...
    - row[0] = Draw Number
    - row[1] = Date
    - row[2:8] = Winning Numbers (6 numbers)
    - row[8] = Additional Number (Bonus)
    """
    # Check primary CSV file first, then fallback to alternative
    toto_csv = str(config.TOTO_CSV_FILE)
    
    if not os.path.exists(toto_csv):
        # Try alternative file (ToTo.csv in project root)
        if hasattr(config, 'TOTO_ALT_CSV_FILE'):
            alt_csv = str(config.TOTO_ALT_CSV_FILE)
            if os.path.exists(alt_csv):
                toto_csv = alt_csv
                logger.info(f"Using alternative TOTO CSV: {toto_csv}")
            else:
                logger.warning(f"TOTO CSV not found: {toto_csv} or {alt_csv}")
                return []
        else:
            logger.warning(f"TOTO CSV not found: {toto_csv}")
            return []
    
    data: List[MutableMapping[str, Any]] = []
    try:
        with open(toto_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            
            for row in reader:
                # Need at least 9 columns: Draw, Date, 6 Numbers, Additional
                if len(row) < 9:
                    continue
                
                # Skip empty rows
                if not row[0] or not row[1]:
                    continue
                
                # Date Parsing - Date is in row[1], not row[0]
                dt = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
                    try:
                        dt = datetime.strptime(row[1], fmt).date()
                        break
                    except ValueError:
                        continue
                
                if dt is None:
                    continue
                
                try:
                    # Numbers are in columns 2-7 (6 numbers)
                    nums = list(map(int, row[2:8]))
                    if len(nums) != 6:
                        continue
                    
                    # Additional number is in column 8
                    bonus = int(row[8]) if row[8] and row[8].isdigit() else 0
                    
                    # Draw number is in column 0
                    draw_num = int(row[0]) if row[0].isdigit() else 0
                    
                    data.append({
                        "date": dt,
                        "nums": nums,
                        "bonus": bonus,
                        "draw": draw_num
                    })
                except (ValueError, IndexError):
                    continue
    
    except Exception as e:
        logger.error(f"Error loading TOTO history: {e}")
        return []
    
    data.sort(key=lambda x: x["date"])
    logger.info(f"Loaded {len(data)} TOTO draws")
    return data


def load_toto_history_for_ai() -> List[MutableMapping[str, Any]]:
    """Alias for AI Engine compatibility."""
    return load_toto_history()


# --- 4D FUNCTIONS ---

def load_4d_history() -> List[MutableMapping[str, Any]]:
    """
    Load 4D history from CSV.
    
    ✅ Safe: Returns [] if file missing (instead of crashing)
    """
    four_d_csv = str(config.FOUR_D_CSV_FILE)
    
    # Graceful handling: return [] if missing
    if not hasattr(config, 'FOUR_D_CSV_FILE') or not os.path.exists(four_d_csv):
        logger.debug(f"4D CSV not found: {four_d_csv} (will use empty history)")
        return []
    
    data: List[MutableMapping[str, Any]] = []
    try:
        with open(four_d_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            
            for row in reader:
                if len(row) < 2:
                    continue
                
                try:
                    draw_date = row[0]
                    draw_number = row[1]
                    prize1 = row[2] if len(row) > 2 else "No"
                    prize2 = row[3] if len(row) > 3 else "No"
                    prize3 = row[4] if len(row) > 4 else "No"
                    
                    data.append({
                        "date": draw_date,
                        "draw_number": draw_number,
                        "prize1": prize1,
                        "prize2": prize2,
                        "prize3": prize3
                    })
                except (ValueError, IndexError):
                    continue
    
    except Exception as e:
        logger.warning(f"Error loading 4D history: {e}")
        return []
    
    logger.info(f"Loaded {len(data)} 4D draws")
    return data
