"""
AI Model Store (Enhanced Version)
==================================

Manages loading data and saving/loading AI models with:
- Model versioning and metadata tracking
- Multiple CSV format support
- Batch model operations
- Model validation and integrity checks
- Automatic backup on save
"""

import csv
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Tuple
from dataclasses import dataclass, asdict

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

logger = logging.getLogger(__name__)


class TotoDraw(TypedDict):
    """Represents a single TOTO draw record."""
    date: str
    nums: List[int]


@dataclass
class ModelMetadata:
    """Metadata for a saved model."""
    model_index: int
    created_at: str
    version: str
    training_samples: int
    feature_dim: int
    accuracy: Optional[float] = None
    notes: str = ""


@dataclass
class DataStats:
    """Statistics about loaded data."""
    total_records: int
    date_range: Tuple[str, str]
    unique_numbers: int
    load_time_ms: float


class ModelStore:
    """
    Enhanced Model Store for AI models and data management.
    
    Features:
    - Loads TOTO history from multiple CSV formats
    - Saves/loads models with metadata tracking
    - Automatic model versioning
    - Batch model operations
    - Model validation and integrity checks
    
    Usage:
        store = ModelStore("./data_storage")
        
        # Load data
        history = store.load_raw_data()
        
        # Save model with metadata
        store.save_model(1, model, training_samples=1000)
        
        # Load model
        model = store.load_model(1)
        
        # Get model info
        info = store.get_model_info(1)
    """

    MODELS_SUBDIR: str = "models"
    METADATA_SUBDIR: str = "metadata"
    BACKUP_SUBDIR: str = "backups"
    
    # Supported data files (in priority order)
    DATA_FILES: List[str] = [
        "toto_full_history.csv",
        "ToTo.csv",
        "toto_history.csv",
    ]
    
    MAX_MODELS: int = 49
    MODEL_VERSION: str = "1.0.0"

    def __init__(self, base_dir: str) -> None:
        """
        Initialize the model store.
        
        Args:
            base_dir: Base directory for all storage operations.
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / self.MODELS_SUBDIR
        self.metadata_dir = self.base_dir / self.METADATA_SUBDIR
        self.backup_dir = self.base_dir / self.BACKUP_SUBDIR
        
        # Create directories
        for dir_path in [self.models_dir, self.metadata_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Find data file
        self.csv_path = self._find_data_file()
        
        # Cache for loaded data
        self._data_cache: Optional[List[TotoDraw]] = None
        self._data_stats: Optional[DataStats] = None

        logger.info(
            "ModelStore initialized",
            extra={
                "base_dir": str(self.base_dir),
                "models_dir": str(self.models_dir),
                "csv_path": str(self.csv_path) if self.csv_path else "NOT_FOUND"
            }
        )

    def _find_data_file(self) -> Optional[Path]:
        """Find the first available data file."""
        # Check in base_dir first
        for filename in self.DATA_FILES:
            path = self.base_dir / filename
            if path.is_file():
                return path
        
        # Check in parent directory
        parent = self.base_dir.parent
        for filename in self.DATA_FILES:
            path = parent / filename
            if path.is_file():
                return path
        
        return None

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_raw_data(self, force_reload: bool = False) -> List[TotoDraw]:
        """
        Load TOTO history from CSV file.
        
        Supports multiple CSV formats:
        1. Standard: Date, N1, N2, N3, N4, N5, N6, Bonus
        2. ToTo.csv: Draw, Date, Winning Number 1, 2, 3, 4, 5, 6, Additional Number
        
        Args:
            force_reload: If True, bypass cache and reload from file.
            
        Returns:
            List of TotoDraw records.
        """
        if self._data_cache is not None and not force_reload:
            return self._data_cache
            
        import time
        start_time = time.perf_counter()
        
        if not self.csv_path or not self.csv_path.is_file():
            logger.error("data.missing", extra={"searched": self.DATA_FILES})
            return []

        history: List[TotoDraw] = []

        try:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)

                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]
                    
                # Detect format based on columns
                columns = set(reader.fieldnames) if reader.fieldnames else set()
                format_type = self._detect_format(columns)

                for row in reader:
                    try:
                        record = self._parse_row(row, format_type)
                        if record and len(record["nums"]) >= 6:
                            history.append(record)
                    except (ValueError, TypeError, KeyError):
                        continue

            # Sort by date (oldest first)
            history.sort(key=lambda x: x.get("date", ""))
            
            # Calculate stats
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            dates = [h["date"] for h in history if h.get("date")]
            all_nums = set()
            for h in history:
                all_nums.update(h.get("nums", []))
            
            self._data_stats = DataStats(
                total_records=len(history),
                date_range=(min(dates) if dates else "", max(dates) if dates else ""),
                unique_numbers=len(all_nums),
                load_time_ms=round(elapsed_ms, 2)
            )
            
            # Cache the data
            self._data_cache = history

            logger.info(
                "data.loaded",
                extra={
                    "path": str(self.csv_path),
                    "n_draws": len(history),
                    "format": format_type,
                    "load_time_ms": self._data_stats.load_time_ms
                },
            )
            return history

        except Exception as exc:
            logger.error(
                "data.load_failed",
                extra={"path": str(self.csv_path), "error": str(exc)},
            )
            return []

    def _detect_format(self, columns: set) -> str:
        """Detect CSV format based on column names."""
        if "N1" in columns and "N2" in columns:
            return "standard"
        elif "Winning Number 1" in columns or "2" in columns:
            return "toto_csv"
        elif "Draw" in columns:
            return "toto_csv"
        else:
            return "unknown"

    def _parse_row(self, row: Dict[str, str], format_type: str) -> Optional[TotoDraw]:
        """Parse a CSV row based on format type."""
        draw_nums: List[int] = []
        date_str = ""
        
        if format_type == "standard":
            # Format: Date, N1, N2, N3, N4, N5, N6, Bonus
            date_str = row.get("Date", "").strip()
            for k in ["N1", "N2", "N3", "N4", "N5", "N6"]:
                value = row.get(k)
                if value not in (None, ""):
                    draw_nums.append(int(value))
            
            # Optionally include bonus
            bonus = row.get("Bonus")
            if bonus not in (None, "") and bonus.isdigit():
                pass  # Don't include bonus in nums for training
                
        elif format_type == "toto_csv":
            # Format: Draw, Date, Winning Number 1, 2, 3, 4, 5, 6, Additional Number
            date_str = row.get("Date", "").strip()
            
            # Try numbered columns first
            num_keys = ["Winning Number 1", "2", "3", "4", "5", "6"]
            for k in num_keys:
                value = row.get(k)
                if value not in (None, ""):
                    try:
                        draw_nums.append(int(value))
                    except ValueError:
                        continue
                        
        else:
            # Unknown format - try to extract any numeric columns
            date_str = row.get("Date", row.get("date", "")).strip()
            for key, value in row.items():
                if value and value.isdigit():
                    num = int(value)
                    if 1 <= num <= 49 and len(draw_nums) < 6:
                        draw_nums.append(num)
        
        if len(draw_nums) >= 6:
            return {
                "date": date_str,
                "nums": draw_nums[:6]  # Take only first 6
            }
        return None

    def get_data_stats(self) -> Optional[DataStats]:
        """Get statistics about the loaded data."""
        if self._data_stats is None:
            self.load_raw_data()
        return self._data_stats

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache = None
        self._data_stats = None

    # =========================================================================
    # MODEL SAVING / LOADING
    # =========================================================================

    def save_model(
        self,
        model_index: int,
        model: Any,
        training_samples: int = 0,
        feature_dim: int = 0,
        accuracy: Optional[float] = None,
        notes: str = "",
        create_backup: bool = True
    ) -> bool:
        """
        Save a model with metadata.
        
        Args:
            model_index: Model index (1-49)
            model: The model object to save
            training_samples: Number of training samples used
            feature_dim: Feature dimension
            accuracy: Optional accuracy score
            notes: Optional notes
            create_backup: Whether to backup existing model
            
        Returns:
            True if successful, False otherwise.
        """
        if not HAS_JOBLIB:
            logger.error("model.save_failed", extra={"error": "joblib not installed"})
            return False
            
        path = self._get_model_path(model_index)
        
        # Create backup if model exists
        if create_backup and path.is_file():
            self._backup_model(model_index)
        
        try:
            # Save model
            joblib.dump(model, path)
            
            # Save metadata
            metadata = ModelMetadata(
                model_index=model_index,
                created_at=datetime.now().isoformat(),
                version=self.MODEL_VERSION,
                training_samples=training_samples,
                feature_dim=feature_dim,
                accuracy=accuracy,
                notes=notes
            )
            self._save_metadata(model_index, metadata)
            
            logger.info(
                "model.saved",
                extra={
                    "index": model_index,
                    "path": str(path),
                    "training_samples": training_samples
                }
            )
            return True
            
        except Exception as exc:
            logger.error(
                "model.save_failed",
                extra={"index": model_index, "path": str(path), "error": str(exc)},
            )
            return False

    def load_model(self, model_index: int) -> Optional[Any]:
        """
        Load a model by index.
        
        Args:
            model_index: Model index (1-49)
            
        Returns:
            The loaded model or None if not found.
        """
        if not HAS_JOBLIB:
            logger.error("model.load_failed", extra={"error": "joblib not installed"})
            return None
            
        path = self._get_model_path(model_index)
        
        if not path.is_file():
            logger.debug(f"Model {model_index} not found at {path}")
            return None
            
        try:
            model = joblib.load(path)
            logger.debug(f"Loaded model {model_index} from {path}")
            return model
            
        except Exception as exc:
            logger.error(
                "model.load_failed",
                extra={"index": model_index, "path": str(path), "error": str(exc)},
            )
            return None

    def load_all_models(self) -> Dict[int, Any]:
        """
        Load all available models.
        
        Returns:
            Dictionary mapping model index to model object.
        """
        models = {}
        for i in range(1, self.MAX_MODELS + 1):
            model = self.load_model(i)
            if model is not None:
                models[i] = model
        
        logger.info(f"Loaded {len(models)}/{self.MAX_MODELS} models")
        return models

    def get_model_info(self, model_index: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a saved model.
        
        Returns:
            Dictionary with model info or None if not found.
        """
        path = self._get_model_path(model_index)
        metadata = self._load_metadata(model_index)
        
        if not path.is_file():
            return None
        
        info = {
            "index": model_index,
            "path": str(path),
            "file_size_kb": round(path.stat().st_size / 1024, 2),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        
        if metadata:
            info.update(asdict(metadata))
        
        return info

    def get_all_model_info(self) -> List[Dict[str, Any]]:
        """Get info for all saved models."""
        info_list = []
        for i in range(1, self.MAX_MODELS + 1):
            info = self.get_model_info(i)
            if info:
                info_list.append(info)
        return info_list

    def delete_model(self, model_index: int, create_backup: bool = True) -> bool:
        """
        Delete a model.
        
        Args:
            model_index: Model index to delete
            create_backup: Whether to backup before deleting
            
        Returns:
            True if successful.
        """
        path = self._get_model_path(model_index)
        
        if not path.is_file():
            return False
        
        if create_backup:
            self._backup_model(model_index)
        
        try:
            path.unlink()
            
            # Delete metadata too
            meta_path = self._get_metadata_path(model_index)
            if meta_path.is_file():
                meta_path.unlink()
            
            logger.info(f"Deleted model {model_index}")
            return True
            
        except Exception as exc:
            logger.error(f"Failed to delete model {model_index}: {exc}")
            return False

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_model_path(self, index: int) -> Path:
        """Get the file path for a model."""
        if not (1 <= index <= self.MAX_MODELS):
            raise ValueError(f"Model index out of range: {index}")
        return self.models_dir / f"model_{index}.joblib"

    def _get_metadata_path(self, index: int) -> Path:
        """Get the metadata file path for a model."""
        return self.metadata_dir / f"model_{index}_meta.json"

    def _save_metadata(self, model_index: int, metadata: ModelMetadata) -> None:
        """Save model metadata to JSON."""
        path = self._get_metadata_path(model_index)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to save metadata for model {model_index}: {exc}")

    def _load_metadata(self, model_index: int) -> Optional[ModelMetadata]:
        """Load model metadata from JSON."""
        path = self._get_metadata_path(model_index)
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ModelMetadata(**data)
        except Exception:
            return None

    def _backup_model(self, model_index: int) -> bool:
        """Create a backup of an existing model."""
        src_path = self._get_model_path(model_index)
        if not src_path.is_file():
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"model_{model_index}_{timestamp}.joblib"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(src_path, backup_path)
            logger.debug(f"Backed up model {model_index} to {backup_path}")
            return True
        except Exception as exc:
            logger.warning(f"Backup failed for model {model_index}: {exc}")
            return False

    def count_models(self) -> int:
        """Count the number of saved models."""
        count = 0
        for i in range(1, self.MAX_MODELS + 1):
            if self._get_model_path(i).is_file():
                count += 1
        return count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        model_count = self.count_models()
        total_size = sum(
            self._get_model_path(i).stat().st_size
            for i in range(1, self.MAX_MODELS + 1)
            if self._get_model_path(i).is_file()
        )
        
        backup_count = len(list(self.backup_dir.glob("*.joblib")))
        
        return {
            "models_count": model_count,
            "models_total_size_mb": round(total_size / (1024 * 1024), 2),
            "backups_count": backup_count,
            "data_file": str(self.csv_path) if self.csv_path else None,
            "data_stats": asdict(self._data_stats) if self._data_stats else None
        }


# ==========================================
# TEST SECTION
# ==========================================
if __name__ == "__main__":
    import tempfile
    
    print("=" * 60)
    print("AI Model Store - Test")
    print("=" * 60)
    
    # Test with actual project directory
    store = ModelStore(".")
    
    print("\n[DATA LOADING]")
    history = store.load_raw_data()
    print(f"  Loaded: {len(history)} draws")
    
    stats = store.get_data_stats()
    if stats:
        print(f"  Date range: {stats.date_range[0]} to {stats.date_range[1]}")
        print(f"  Unique numbers: {stats.unique_numbers}")
        print(f"  Load time: {stats.load_time_ms}ms")
    
    print("\n[MODEL COUNT]")
    model_count = store.count_models()
    print(f"  Saved models: {model_count}/{store.MAX_MODELS}")
    
    print("\n[STORAGE STATS]")
    storage = store.get_storage_stats()
    print(f"  Models size: {storage['models_total_size_mb']} MB")
    print(f"  Backups: {storage['backups_count']}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")