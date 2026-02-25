# =============================================================================
# SG LOTTERY AI — MASTER SUITE (PROFESSIONAL EDITION)
# UI: UNCHANGED | ENGINE: INTEGRATED WITH REAL DATA
# =============================================================================
from __future__ import annotations

import csv
import logging
import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from typing import Optional, List, Dict, Tuple, Any

# ---------------------------------------------------------
# MODULE IMPORTS - INTEGRATED WITH UPGRADED MODULES
# ---------------------------------------------------------

# Core modules availability flags
HAS_CONFIG = False
HAS_SCRAPER = False
HAS_AI_ENGINE = False
HAS_AI_TRAINER = False
HAS_DATA_MANAGER = False

# Try to import upgraded modules
try:
    import config
    HAS_CONFIG = True
except ImportError as e:
    print(f"[WARNING] config module not found: {e}")

try:
    from scraper import TotoScraper, get_latest_result_from_csv, get_all_results_from_csv, update_toto
    HAS_SCRAPER = True
except ImportError as e:
    print(f"[WARNING] scraper module not found: {e}")

try:
    from ai_engine import TotoBrain, PredictionStrategy
    HAS_AI_ENGINE = True
except ImportError as e:
    print(f"[WARNING] ai_engine module not found: {e}")

try:
    from ai_trainer import TotoTrainer, TrainingMode, TrainingSession
    HAS_AI_TRAINER = True
except ImportError as e:
    print(f"[WARNING] ai_trainer module not found: {e}")

try:
    import data_manager
    HAS_DATA_MANAGER = True
except ImportError as e:
    print(f"[WARNING] data_manager module not found: {e}")

# Plotting libraries
HAS_PLOTTING = False
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.cluster import KMeans
    HAS_PLOTTING = True
except ImportError:
    pass

# Pillow for images
HAS_PILLOW = False
try:
    from PIL import Image, ImageTk
    HAS_PILLOW = True
except ImportError:
    pass

# ---------------------------------------------------------
# LOGGING & CONFIG
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Colors (Original UI Colors - UNCHANGED)
BG_COLOR = "#4a0000"
GOLD = "#ffd700"
RED = "#b71c1c"
BLUE = "#1e3799"

# Path Setup
BASE_DIR = Path(__file__).parent.resolve()

# CSV file path
if HAS_CONFIG:
    CSV_FILE = str(config.TOTO_CSV_FILE) if hasattr(config, 'TOTO_CSV_FILE') else str(BASE_DIR / "ToTo.csv")
else:
    CSV_FILE = str(BASE_DIR / "ToTo.csv")


# ---------------------------------------------------------
# INTERNAL LOGIC - INTEGRATED WITH REAL DATA
# ---------------------------------------------------------

def internal_robust_update() -> Tuple[bool, str]:
    """
    Real Data Update using the upgraded scraper module.
    Fetches latest TOTO results from Singapore Pools.
    """
    try:
        if HAS_SCRAPER:
            # Use the upgraded scraper
            success, msg = update_toto()
            return success, msg
        else:
            # Fallback: just check local data
            time.sleep(0.5)
            if os.path.exists(CSV_FILE):
                return True, "Using local data (scraper not available)"
            return False, "No data source available"
            
    except Exception as e:
        logger.error(f"Update error: {e}")
        return False, str(e)


class InternalBrain:
    """
    Wrapper class connecting UI to the upgraded AI Engine (TotoBrain).
    Uses frequency analysis and weighted prediction strategies.
    """
    
    def __init__(self):
        self.brain: Optional[Any] = None
        self._initialize_brain()
    
    def _initialize_brain(self) -> None:
        """Initialize the AI brain."""
        if HAS_AI_ENGINE:
            try:
                self.brain = TotoBrain()
                logger.info("AI Engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI Engine: {e}")
                self.brain = None
        else:
            logger.warning("AI Engine not available - using random fallback")
    
    def reload(self) -> None:
        """Reload data and models."""
        if self.brain and hasattr(self.brain, 'reload_data'):
            self.brain.reload_data()
        elif HAS_AI_ENGINE:
            self._initialize_brain()
    
    def predict(self, strategy: str = "ensemble") -> Tuple[List[int], float]:
        """
        Generate prediction using the AI Engine.
        
        Args:
            strategy: Prediction strategy (ensemble, hot, cold, balanced)
            
        Returns:
            Tuple of (6 predicted numbers, confidence score)
        """
        if not self.brain:
            # Fallback to random
            nums = sorted(random.sample(range(1, 50), 6))
            return nums, 0.1
        
        try:
            # Map strategy string to enum
            strat_map = {
                "ensemble": PredictionStrategy.ENSEMBLE,
                "hot": PredictionStrategy.HOT_NUMBERS,
                "cold": PredictionStrategy.COLD_NUMBERS,
                "balanced": PredictionStrategy.BALANCED,
            }
            strat = strat_map.get(strategy, PredictionStrategy.ENSEMBLE)
            
            result = self.brain.predict(strat)
            return result.numbers, result.confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return sorted(random.sample(range(1, 50), 6)), 0.1
    
    def get_hot_numbers(self, count: int = 10) -> List[Tuple[int, int]]:
        """Get hot numbers from history."""
        if self.brain and hasattr(self.brain, 'get_hot_numbers'):
            return self.brain.get_hot_numbers(count)
        return []
    
    def get_cold_numbers(self, count: int = 10) -> List[Tuple[int, int]]:
        """Get cold numbers from history."""
        if self.brain and hasattr(self.brain, 'get_cold_numbers'):
            return self.brain.get_cold_numbers(count)
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics summary."""
        if self.brain and hasattr(self.brain, 'get_statistics_summary'):
            return self.brain.get_statistics_summary()
        return {"error": "Statistics not available"}


# ---------------------------------------------------------
# UI APPLICATION (DESIGN UNCHANGED)
# ---------------------------------------------------------

@dataclass(frozen=True)
class UIUpdateMessage:
    """Message for thread-safe UI updates."""
    content: str
    level: str = "info"
    done: bool = False


class App(tk.Tk):
    """Main Application Window - UI Design UNCHANGED."""
    
    QUEUE_POLL_MS = 100
    QUEUE_MAX_PER_TICK = 50

    def __init__(self) -> None:
        super().__init__()
        self.title("SG Toto AI")
        self.geometry("1180x950")
        self.configure(bg=BG_COLOR)
        
        # State
        self._closing: bool = False
        self.queue: "queue.Queue[UIUpdateMessage]" = queue.Queue()
        
        # Initialize the AI Brain
        self.brain = InternalBrain()
        
        # Image references
        self.img_ref_left: Optional[tk.PhotoImage] = None
        self.img_ref_right: Optional[tk.PhotoImage] = None

        self._setup_styles()
        self._build_layout()
        self.after(100, self._check_queue)
        self.load_history_table()
        
        # Show status on startup
        self._show_startup_status()

    def _show_startup_status(self) -> None:
        """Show module status on startup."""
        self.log("=" * 50)
        self.log("SG TOTO AI - SYSTEM STATUS")
        self.log("=" * 50)
        self.log(f"  Config Module: {'OK' if HAS_CONFIG else 'Missing'}")
        self.log(f"  Scraper Module: {'OK' if HAS_SCRAPER else 'Missing'}")
        self.log(f"  AI Engine: {'OK' if HAS_AI_ENGINE else 'Missing'}")
        self.log(f"  AI Trainer: {'OK' if HAS_AI_TRAINER else 'Missing'}")
        self.log(f"  Plotting: {'OK' if HAS_PLOTTING else 'Missing'}")
        
        if self.brain.brain:
            stats = self.brain.get_statistics()
            if "total_draws" in stats:
                self.log(f"\n  Data Loaded: {stats['total_draws']} draws")
                if "date_range" in stats:
                    self.log(f"  Date Range: {stats['date_range']['oldest']} to {stats['date_range']['newest']}")
        
        self.log("=" * 50)
        self.log("Ready! Click 'Run AI' to get predictions.\n")

    def _setup_styles(self) -> None:
        """Setup ttk styles - UNCHANGED."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except:
            pass
        style.configure("Gold.TButton", font=("Segoe UI", 11, "bold"), padding=10, foreground="black", background=GOLD)
        style.configure("Red.TButton", font=("Segoe UI", 10, "bold"), padding=8, foreground="white", background=RED)
        style.configure("Blue.TButton", font=("Segoe UI", 10, "bold"), padding=8, foreground="white", background=BLUE)
        style.configure("Treeview", font=("Consolas", 10), rowheight=25, background="#2D2D2D", fieldbackground="#2D2D2D", foreground="white")
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"), background="#800000", foreground="white")

    def _safe_load_image(self, name: str, target_h: int = 90) -> Optional[tk.PhotoImage]:
        """Load image from assets folder - with improved aspect ratio handling."""
        if not HAS_PILLOW:
            return None
        try:
            base_path = Path(__file__).parent.resolve()
            if getattr(sys, "frozen", False):
                base_path = Path(sys.executable).parent
            assets = base_path / "assets"
            
            for ext in ["png", "jpg", "jpeg"]:
                path = assets / f"{name}.{ext}"
                if path.exists():
                    img = Image.open(path)
                    orig_w, orig_h = img.size
                    
                    # Calculate aspect ratio
                    aspect_ratio = orig_w / orig_h
                    
                    # For very tall images (aspect ratio < 0.5), use width-based scaling
                    # to prevent images from becoming too narrow
                    if aspect_ratio < 0.5:
                        # Use a minimum width of 80px for narrow images
                        target_w = max(80, int(target_h * aspect_ratio))
                        new_h = int(target_w / aspect_ratio)
                        resized = img.resize((target_w, new_h), Image.Resampling.LANCZOS)
                    else:
                        # Normal height-based scaling
                        ratio = target_h / orig_h
                        resized = img.resize((int(orig_w * ratio), target_h), Image.Resampling.LANCZOS)
                    
                    return ImageTk.PhotoImage(resized)
            return None
        except Exception as e:
            logger.warning(f"Failed to load image {name}: {e}")
            return None

    def _build_layout(self) -> None:
        """Build the UI layout - UNCHANGED DESIGN."""
        # Header
        header = tk.Frame(self, bg="#6d0000", pady=10)
        header.pack(fill="x")

        self.img_ref_left = self._safe_load_image("logo", 90)
        if self.img_ref_left:
            tk.Label(header, image=self.img_ref_left, bg="#6d0000").pack(side=tk.LEFT, padx=20)

        title_canvas = tk.Canvas(header, bg="#6d0000", height=85, width=650, highlightthickness=0)
        title_canvas.pack(side=tk.LEFT, expand=True)
        title_canvas.create_text(328, 38, text="SG TOTO AI", font=("Impact", 40), fill="black")
        title_canvas.create_text(325, 35, text="SG TOTO AI", font=("Impact", 40), fill="#FFD700")
        title_canvas.create_text(327, 72, text="PREDICTION ENGINE", font=("Arial", 14, "bold"), fill="black")
        title_canvas.create_text(325, 70, text="PREDICTION ENGINE", font=("Arial", 14, "bold"), fill="white")

        self.img_ref_right = self._safe_load_image("logo2", 90)
        if self.img_ref_right:
            tk.Label(header, image=self.img_ref_right, bg="#6d0000").pack(side=tk.RIGHT, padx=20)

        # Button Panel
        panel = tk.Frame(self, bg=BG_COLOR, pady=15)
        panel.pack(fill="x")

        self._btn_run_ai = ttk.Button(panel, text="Run AI", style="Gold.TButton", width=15, command=self.on_spin)
        self._btn_run_ai.pack(side=tk.LEFT, padx=10)
        self._btn_simulate = ttk.Button(panel, text="Retrain AI", style="Blue.TButton", width=15, command=self.on_retrain)
        self._btn_simulate.pack(side=tk.LEFT, padx=5)
        self._btn_update = ttk.Button(panel, text="Update", style="Blue.TButton", width=15, command=self.on_update)
        self._btn_update.pack(side=tk.LEFT, padx=5)
        self._btn_tickets = ttk.Button(panel, text="Tickets", style="Red.TButton", width=20, command=self.on_gen_tickets)
        self._btn_tickets.pack(side=tk.LEFT, padx=20)
        self._btn_save_log = ttk.Button(panel, text="SAVE TO DESKTOP", style="Red.TButton", width=20, command=self.on_save_log)
        self._btn_save_log.pack(side=tk.LEFT, padx=5)

        # Analytics Panel
        analytics = tk.Frame(self, bg=BG_COLOR, pady=10)
        analytics.pack(fill="x")
        tk.Label(analytics, text="Analytics:", font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=GOLD).pack(side=tk.LEFT, padx=10)

        for txt, cmd in [
            ("Heatmap", self.on_heatmap),
            ("Models", self.on_model_compare),
            ("Clusters", self.on_cluster),
            ("Gravity", self.on_gravity),
            ("Patterns", self.on_pattern_analysis),
            ("System Bet", self.on_system_bet)
        ]:
            ttk.Button(analytics, text=txt, style="Red.TButton", command=cmd).pack(side=tk.LEFT, padx=4)

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_main = tk.Frame(self.notebook, bg="#1a0000")
        self.tab_history = tk.Frame(self.notebook, bg="#2D2D2D")
        self.tab_4d = tk.Frame(self.notebook, bg="#1a0000")

        self.notebook.add(self.tab_main, text=" Prediction Log ")
        self.notebook.add(self.tab_history, text=" Full History ")
        self.notebook.add(self.tab_4d, text=" 4D Analysis ")

        # Main log area
        self.text = scrolledtext.ScrolledText(self.tab_main, font=("Consolas", 11), bg="#1a0000", fg="#f1f2f6")
        self.text.pack(fill="both", expand=True, padx=5, pady=5)

        self.setup_history_tab()
        self.setup_4d_tab()

        # Disclaimer at bottom
        disclaimer = tk.Label(self, text="This app is not affiliated with Singapore Pools", 
                             font=("Segoe UI", 9, "italic"), bg=BG_COLOR, fg="#aaaaaa")
        disclaimer.pack(side=tk.BOTTOM, pady=5)

    def _toggle_buttons(self, enabled: bool) -> None:
        """Enable/disable buttons during operations."""
        state = "normal" if enabled else "disabled"
        for btn in (self._btn_run_ai, self._btn_simulate, self._btn_update, self._btn_tickets, self._btn_save_log):
            if btn:
                btn.configure(state=state)

    def log(self, msg: str) -> None:
        """Log message to the text area."""
        try:
            self.text.insert(tk.END, msg + "\n")
            self.text.see(tk.END)
        except:
            pass

    def _check_queue(self) -> None:
        """Check message queue for thread-safe updates."""
        processed = 0
        while processed < self.QUEUE_MAX_PER_TICK:
            try:
                item = self.queue.get_nowait()
                if item.level == "error":
                    self.log(f"[ERROR] {item.content}")
                elif item.level == "warning":
                    self.log(f"[WARN] {item.content}")
                else:
                    self.log(item.content)
                if item.done:
                    self._toggle_buttons(True)
                processed += 1
            except queue.Empty:
                break
        if not self._closing:
            self.after(self.QUEUE_POLL_MS, self._check_queue)

    def _run_in_thread(self, name: str, fn) -> None:
        """Run function in background thread."""
        def worker() -> None:
            try:
                fn()
            except Exception as e:
                logger.exception(f"Worker {name} crashed")
                self.queue.put(UIUpdateMessage(content=f"Error in {name}: {e}", level="error", done=True))
        threading.Thread(target=worker, name=name, daemon=True).start()

    def on_closing(self) -> None:
        """Handle window close."""
        self._closing = True
        try:
            if HAS_PLOTTING:
                plt.close("all")
            self.destroy()
        except:
            pass

    # =========================================================
    # UPDATE - Uses Real Scraper
    # =========================================================
    def on_update(self) -> None:
        """Update data from Singapore Pools."""
        self.log("\n[UPDATE] Fetching latest data...")
        self._toggle_buttons(False)
        self._run_in_thread("update", self._update_thread)

    def _update_thread(self) -> None:
        """Background thread for updating data."""
        ok, msg = internal_robust_update()
        if ok:
            self.queue.put(UIUpdateMessage(content=f"[OK] {msg}"))
            self.brain.reload()
            self.after(0, self.load_history_table)
            self.queue.put(UIUpdateMessage(content="[OK] Data sync complete", done=True))
        else:
            self.queue.put(UIUpdateMessage(content=f"[ERROR] {msg}", level="error", done=True))

    # =========================================================
    # RUN AI - Uses Real AI Engine
    # =========================================================
    def on_spin(self) -> None:
        """Run AI prediction."""
        self.log("\n[AI] Running prediction engine...")
        self._toggle_buttons(False)
        self._run_in_thread("predict", self._spin_thread)

    def _spin_thread(self) -> None:
        """Background thread for AI prediction."""
        try:
            self.brain.reload()
            
            # Get predictions with different strategies
            self.queue.put(UIUpdateMessage(content="\n" + "=" * 40))
            self.queue.put(UIUpdateMessage(content="      AI PREDICTION RESULTS"))
            self.queue.put(UIUpdateMessage(content="=" * 40))
            
            # Ensemble prediction (main)
            nums, conf = self.brain.predict("ensemble")
            self.queue.put(UIUpdateMessage(content=f"\n  RECOMMENDED NUMBERS:"))
            self.queue.put(UIUpdateMessage(content=f"  >>> {nums} <<<"))
            self.queue.put(UIUpdateMessage(content=f"  Confidence: {conf*100:.1f}%"))
            
            # Additional strategies
            hot_nums, hot_conf = self.brain.predict("hot")
            cold_nums, cold_conf = self.brain.predict("cold")
            
            self.queue.put(UIUpdateMessage(content=f"\n  Alternative Picks:"))
            self.queue.put(UIUpdateMessage(content=f"    Hot Strategy: {hot_nums}"))
            self.queue.put(UIUpdateMessage(content=f"    Cold Strategy: {cold_nums}"))
            
            # Show hot/cold numbers
            hot = self.brain.get_hot_numbers(5)
            cold = self.brain.get_cold_numbers(5)
            
            if hot:
                hot_list = [n for n, _ in hot]
                self.queue.put(UIUpdateMessage(content=f"\n  Hot Numbers: {hot_list}"))
            if cold:
                cold_list = [n for n, _ in cold]
                self.queue.put(UIUpdateMessage(content=f"  Cold Numbers: {cold_list}"))
            
            self.queue.put(UIUpdateMessage(content="\n" + "=" * 40, done=True))
            
        except Exception as e:
            self.queue.put(UIUpdateMessage(content=f"AI Error: {e}", level="error", done=True))

    # =========================================================
    # RETRAIN - Uses Real AI Trainer
    # =========================================================
    def on_retrain(self) -> None:
        """Retrain AI models."""
        if HAS_AI_TRAINER:
            if not messagebox.askyesno("Retrain AI?", "Start AI model training?\nThis may take several minutes."):
                return
            self.log("\n[TRAINING] Starting AI model training...")
            self._toggle_buttons(False)
            self._run_in_thread("train", self._train_thread)
        else:
            messagebox.showinfo("Info", "AI Trainer module not available.\nRunning simulation instead.")
            self._run_simulation()

    def _train_thread(self) -> None:
        """Background thread for training."""
        try:
            if HAS_CONFIG:
                data_dir = str(config.DATA_STORAGE_DIR)
            else:
                data_dir = str(BASE_DIR / "data_storage")
            
            trainer = TotoTrainer(data_dir, mode=TrainingMode.FAST)
            
            # Progress callback
            def on_progress(current: int, total: int) -> None:
                self.queue.put(UIUpdateMessage(content=f"  Training model {current}/{total}..."))
            
            session = trainer.train_all(progress_callback=on_progress)
            
            self.queue.put(UIUpdateMessage(content=f"\n[OK] Training Complete!"))
            self.queue.put(UIUpdateMessage(content=f"  Models trained: {session.models_trained}"))
            self.queue.put(UIUpdateMessage(content=f"  Average accuracy: {session.avg_accuracy*100:.1f}%"))
            self.queue.put(UIUpdateMessage(content=f"  Time: {session.total_time_seconds}s", done=True))
            
            # Reload brain with new models
            self.brain.reload()
            
        except Exception as e:
            self.queue.put(UIUpdateMessage(content=f"Training Failed: {e}", level="error", done=True))

    def _run_simulation(self) -> None:
        """Fallback simulation."""
        self._toggle_buttons(False)
        self._run_in_thread("simulate", self._sim_thread)

    def _sim_thread(self) -> None:
        """Simulation thread."""
        nums = list(range(1, 50))
        results = [tuple(sorted(random.sample(nums, 6))) for _ in range(5000)]
        top = Counter(results).most_common(5)
        msg = "\n".join([f"#{i}: {list(x[0])} ({x[1]} hits)" for i, x in enumerate(top, 1)])
        self.queue.put(UIUpdateMessage(content=f"\n[SIMULATION] Top combinations:\n{msg}", done=True))

    # =========================================================
    # GENERATE TICKETS
    # =========================================================
    def on_gen_tickets(self) -> None:
        """Generate lottery tickets."""
        qty = simpledialog.askinteger("Generate Tickets", "How many tickets?", minvalue=1, maxvalue=100)
        if qty and qty > 0:
            self._toggle_buttons(False)
            self._run_in_thread("tickets", lambda: self._gen_tickets_thread(qty))

    def _gen_tickets_thread(self, qty: int) -> None:
        """Generate tickets in background."""
        tickets = []
        
        # Use AI to generate smarter tickets if available
        if self.brain.brain:
            try:
                results = self.brain.brain.predict_multiple(count=qty)
                tickets = [r.numbers for r in results]
            except:
                pass
        
        # Fallback to random if AI failed
        while len(tickets) < qty:
            tickets.append(sorted(random.sample(range(1, 50), 6)))
        
        self.queue.put(UIUpdateMessage(content=f"\n[TICKETS] Generated {qty} tickets:"))
        
        # Show ALL tickets in the log (for saving)
        for i, t in enumerate(tickets, 1):
            self.queue.put(UIUpdateMessage(content=f"  #{i:02d}: {t}"))
        
        self.queue.put(UIUpdateMessage(content="", done=True))

    # =========================================================
    # HISTORY TAB - Uses Real Data
    # =========================================================
    def setup_history_tab(self) -> None:
        """Setup history table - UNCHANGED DESIGN."""
        ttk.Button(self.tab_history, text="Reload Table", command=self.load_history_table).pack(pady=5, padx=10, anchor="e")
        cols = ("Date", "N1", "N2", "N3", "N4", "N5", "N6", "Bonus")
        self.tree_hist = ttk.Treeview(self.tab_history, columns=cols, show="headings", height=20)
        for c in cols:
            self.tree_hist.heading(c, text=c)
            self.tree_hist.column(c, width=100, anchor="center")
        sb = ttk.Scrollbar(self.tab_history, orient="vertical", command=self.tree_hist.yview)
        self.tree_hist.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree_hist.pack(fill="both", expand=True, padx=10)

    def load_history_table(self) -> None:
        """Load history data into table."""
        # Clear existing
        for i in self.tree_hist.get_children():
            self.tree_hist.delete(i)
        
        # Try to load from scraper module first
        if HAS_SCRAPER:
            try:
                results = get_all_results_from_csv(limit=500)
                for r in results:
                    vals = [r.draw_date] + r.winning_numbers + [r.additional_number]
                    self.tree_hist.insert("", "end", values=vals)
                return
            except Exception as e:
                logger.warning(f"Scraper load failed: {e}")
        
        # Fallback to data_manager
        if HAS_DATA_MANAGER:
            try:
                data = data_manager.load_toto_history()
                for r in reversed(data[-500:]):
                    dt_str = r['date'].strftime("%Y-%m-%d") if hasattr(r['date'], 'strftime') else str(r['date'])
                    vals = [dt_str] + r['nums'] + [r.get('bonus', 0)]
                    self.tree_hist.insert("", "end", values=vals)
                return
            except Exception as e:
                logger.warning(f"Data manager load failed: {e}")
        
        # Final fallback: direct CSV read
        try:
            csv_path = Path(CSV_FILE)
            if not csv_path.exists():
                # Try ToTo.csv in base dir
                csv_path = BASE_DIR / "ToTo.csv"
            
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    rows = list(reader)
                    
                    # Show newest first
                    for row in rows[:500]:
                        if len(row) >= 9:
                            # ToTo.csv format: Draw, Date, N1-N6, Additional
                            vals = [row[1]] + [row[i] for i in range(2, 8)] + [row[8]]
                            self.tree_hist.insert("", "end", values=vals)
        except Exception as e:
            logger.error(f"CSV load failed: {e}")

    # =========================================================
    # 4D TAB - UNCHANGED
    # =========================================================
    def setup_4d_tab(self) -> None:
        """Setup 4D tab - UNCHANGED DESIGN."""
        control = tk.Frame(self.tab_4d, bg="#1a0000", pady=10)
        control.pack(fill="x")
        ttk.Button(control, text="Analyze", style="Blue.TButton", command=self.on_4d_analyze).pack(side=tk.LEFT, padx=10)
        ttk.Button(control, text="Lucky", style="Gold.TButton", command=self.on_4d_lucky).pack(side=tk.LEFT, padx=10)
        self.lbl_4d = tk.Label(self.tab_4d, text="----", font=("Consolas", 60, "bold"), bg="#1a0000", fg=GOLD)
        self.lbl_4d.pack(pady=30)
        self.log_4d = scrolledtext.ScrolledText(self.tab_4d, font=("Consolas", 12), bg="#1E1E1E", fg="#00FF00", height=12)
        self.log_4d.pack(fill="both", expand=True, padx=10, pady=10)

    def on_4d_analyze(self) -> None:
        """4D analysis."""
        self.log_4d.delete("1.0", tk.END)
        self.log_4d.insert(tk.END, "Analyzing 4D Patterns...\n")
        self.log_4d.insert(tk.END, "Trend Analysis: NEUTRAL\n")
        self.log_4d.insert(tk.END, "Hot Patterns: 12xx, 99xx\n")

    def on_4d_lucky(self) -> None:
        """Generate lucky 4D number."""
        self.lbl_4d.config(text=f"{random.randint(0, 9999):04d}", fg="#00d2d3")

    # =========================================================
    # ANALYTICS - Uses Real Data
    # =========================================================
    def on_heatmap(self) -> None:
        """Show frequency heatmap."""
        if not HAS_PLOTTING:
            messagebox.showinfo("Error", "Matplotlib not installed.")
            return
        
        freq = np.zeros(49)
        
        # Load real data
        if HAS_SCRAPER:
            try:
                results = get_all_results_from_csv()
                for r in results:
                    for n in r.winning_numbers:
                        if 1 <= n <= 49:
                            freq[n-1] += 1
            except:
                pass
        elif HAS_DATA_MANAGER:
            try:
                data = data_manager.load_toto_history()
                for r in data:
                    for n in r['nums']:
                        if 1 <= n <= 49:
                            freq[n-1] += 1
            except:
                pass
        
        plt.figure(figsize=(12, 3))
        sns.heatmap([freq], cmap="YlOrRd", annot=False, cbar=True)
        plt.title("TOTO Number Frequency Heatmap (Real Data)")
        plt.xlabel("Number (1-49)")
        plt.xticks(np.arange(0.5, 49.5, 5), range(1, 50, 5))
        plt.yticks([])
        plt.tight_layout()
        plt.show(block=False)

    def on_cluster(self) -> None:
        """Show clustering visualization."""
        if not HAS_PLOTTING:
            return
        
        freq = np.zeros((49, 1))
        
        if HAS_SCRAPER:
            try:
                results = get_all_results_from_csv()
                for r in results:
                    for n in r.winning_numbers:
                        if 1 <= n <= 49:
                            freq[n-1] += 1
            except:
                pass
        
        if freq.sum() == 0:
            messagebox.showinfo("Info", "No data for clustering")
            return
        
        try:
            km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(freq)
            plt.figure(figsize=(12, 5))
            colors = ['red', 'blue', 'green', 'orange']
            for i, label in enumerate(km.labels_):
                plt.bar(i+1, freq[i][0], color=colors[label], alpha=0.7)
            plt.title("Number Clustering by Frequency")
            plt.xlabel("Number")
            plt.ylabel("Frequency")
            plt.show(block=False)
        except Exception as e:
            messagebox.showinfo("Error", f"Clustering failed: {e}")

    def on_gravity(self) -> None:
        """Show gravity analysis."""
        if not HAS_PLOTTING:
            return
        
        # Calculate recency score
        recency = np.zeros(49)
        
        if HAS_SCRAPER:
            try:
                results = get_all_results_from_csv(limit=100)
                for idx, r in enumerate(results):
                    for n in r.winning_numbers:
                        if 1 <= n <= 49:
                            recency[n-1] += 1.0 / (idx + 1)
            except:
                pass
        
        plt.figure(figsize=(12, 5))
        plt.bar(range(1, 50), recency, color='purple', alpha=0.7)
        plt.title("Number Gravity (Recency-Weighted)")
        plt.xlabel("Number")
        plt.ylabel("Gravity Score")
        plt.show(block=False)

    def on_pattern_analysis(self) -> None:
        """Show pattern analysis."""
        if not HAS_PLOTTING:
            return
        
        odd_ratios = []
        low_ratios = []
        
        if HAS_SCRAPER:
            try:
                results = get_all_results_from_csv(limit=100)
                for r in results:
                    nums = r.winning_numbers
                    odd_ratios.append(sum(1 for n in nums if n % 2 == 1) / 6)
                    low_ratios.append(sum(1 for n in nums if n <= 25) / 6)
            except:
                pass
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(odd_ratios, color='blue', alpha=0.7)
        axes[0].axhline(y=0.5, color='red', linestyle='--')
        axes[0].set_title("Odd/Even Ratio Trend")
        axes[0].set_ylabel("Odd Ratio")
        
        axes[1].plot(low_ratios, color='green', alpha=0.7)
        axes[1].axhline(y=0.5, color='red', linestyle='--')
        axes[1].set_title("Low/High Ratio Trend")
        axes[1].set_ylabel("Low (1-25) Ratio")
        
        plt.tight_layout()
        plt.show(block=False)

    # =========================================================
    # SAVE LOG - UNCHANGED
    # =========================================================
    def on_save_log(self) -> None:
        """Save log to desktop."""
        content = self.text.get("1.0", tk.END)
        if not content.strip():
            messagebox.showwarning("Warning", "Log is empty!")
            return
        
        desktop = Path.home() / "Desktop"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"SG_Lottery_Log_{timestamp}.txt"
        full_path = desktop / filename

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write("=== SG TOTO AI PREDICTION LOG ===\n")
                f.write(f"Saved on: {timestamp}\n")
                f.write("=================================\n\n")
                f.write(content)
            
            messagebox.showinfo("SUCCESS!", f"Saved to Desktop!\n\nFile: {filename}")
            
        except Exception as e:
            try:
                with open(BASE_DIR / filename, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Saved Locally", f"Saved in app folder: {filename}")
            except:
                messagebox.showerror("Error", f"Could not save file:\n{str(e)}")

    def on_model_compare(self) -> None:
        """Show model comparison."""
        if HAS_AI_ENGINE and self.brain.brain:
            stats = self.brain.get_statistics()
            msg = f"AI Engine Status: Active\n"
            if "total_draws" in stats:
                msg += f"Training Data: {stats['total_draws']} draws\n"
            msg += f"Strategy: Ensemble (6 methods combined)"
            messagebox.showinfo("AI Models", msg)
        else:
            messagebox.showinfo("AI Models", "AI Engine not loaded")

    def on_system_bet(self) -> None:
        """Show system bet calculator."""
        bet_type = simpledialog.askinteger("System Bet", "Select type (7-12):", minvalue=7, maxvalue=12)
        if bet_type:
            combos = {7: 7, 8: 28, 9: 84, 10: 210, 11: 462, 12: 924}
            cost = combos.get(bet_type, 0)
            messagebox.showinfo("System Bet", f"System {bet_type}\nCombinations: {cost}\nCost: ${cost}")


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()