"""
SG Lottery Suite Configuration (Enhanced Version)
===================================================

Centralized configuration management with:
- Environment variable support
- Multiple data source fallbacks
- Runtime configuration override
- Configuration validation
- Singleton pattern for global access
"""

import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from functools import lru_cache
import json


# ============================================================================
# PATH UTILITIES
# ============================================================================

def get_base_path() -> Path:
    """
    Returns the correct base path whether running as code or .exe
    
    - Frozen (PyInstaller): Returns the folder containing the .exe
    - Normal: Returns the folder containing this config.py file
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE (PyInstaller one-dir mode)
        return Path(sys.executable).parent
    else:
        # Running as Python script
        return Path(__file__).parent.resolve()


def find_data_file(filename: str, search_paths: List[Path]) -> Optional[Path]:
    """
    Search for a data file in multiple locations.
    
    Args:
        filename: Name of the file to find
        search_paths: List of directories to search
        
    Returns:
        Path to file if found, None otherwise
    """
    for base_path in search_paths:
        file_path = base_path / filename
        if file_path.is_file():
            return file_path
    return None


# ============================================================================
# LOGGER SETUP
# ============================================================================

def _setup_config_logger() -> logging.Logger:
    """Set up a dedicated logger for configuration."""
    logger = logging.getLogger("lottery_suite.config")
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


_logger = _setup_config_logger()


# ============================================================================
# CONFIGURATION DATA CLASS
# ============================================================================

@dataclass
class Config:
    """
    Application configuration container.
    
    All paths are stored as Path objects for cross-platform compatibility.
    Settings can be overridden via environment variables (prefix: SG_LOTTERY_)
    """
    
    # Core paths
    project_root: Path
    data_storage_dir: Path
    models_dir: Path
    logs_dir: Path
    
    # Data files
    toto_csv_file: Path
    toto_alt_csv_file: Path  # Alternative TOTO data file
    four_d_csv_file: Path
    
    # TOTO settings
    toto_max_number: int = 49
    toto_pick_count: int = 6
    toto_analysis_window: int = 100
    
    # 4D settings
    four_d_sample_size: int = 300
    four_d_analysis_window: int = 50
    
    # AI settings
    ai_random_state: int = 42
    ai_feature_window: int = 50
    ai_test_split: float = 0.2
    ai_n_estimators: int = 100
    ai_max_depth: int = 4
    
    # Performance settings
    cache_max_age_seconds: int = 3600
    csv_read_retries: int = 3
    csv_read_retry_delay_seconds: float = 0.1
    
    # Runtime flags
    skip_file_validation: bool = False
    debug_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (with paths as strings)."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.
        
        Returns:
            List of warning messages (empty if all valid)
        """
        warnings = []
        
        # Check directories
        if not self.data_storage_dir.exists():
            warnings.append(f"Data directory not found: {self.data_storage_dir}")
        
        # Check data files
        if not self.toto_csv_file.exists() and not self.toto_alt_csv_file.exists():
            warnings.append("No TOTO data file found")
        
        # Validate numeric ranges
        if not 0 < self.ai_test_split < 1.0:
            warnings.append(f"Invalid test split: {self.ai_test_split}")
        
        if self.ai_n_estimators < 1:
            warnings.append(f"Invalid n_estimators: {self.ai_n_estimators}")
        
        return warnings
    
    def get_toto_data_file(self) -> Optional[Path]:
        """Get the first available TOTO data file."""
        if self.toto_csv_file.exists():
            return self.toto_csv_file
        elif self.toto_alt_csv_file.exists():
            return self.toto_alt_csv_file
        return None


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def _get_env_value(key: str, default: Any = None) -> Any:
    """Get environment variable with SG_LOTTERY_ prefix."""
    env_key = f"SG_LOTTERY_{key.upper()}"
    return os.environ.get(env_key, default)


def load_config(
    skip_file_validation: bool = False,
    custom_data_dir: Optional[Path] = None
) -> Config:
    """
    Load and initialize application configuration.
    
    Configuration sources (in order of priority):
    1. Environment variables (SG_LOTTERY_*)
    2. Custom parameters passed to this function
    3. Default values
    
    Args:
        skip_file_validation: Skip checking if files exist
        custom_data_dir: Override the default data directory
        
    Returns:
        Initialized Config object
    """
    base_path = get_base_path()
    
    # Determine data directory
    if custom_data_dir:
        data_dir = Path(custom_data_dir)
    else:
        # Check environment variable first
        env_data_dir = _get_env_value("DATA_DIR")
        if env_data_dir:
            data_dir = Path(env_data_dir)
        else:
            # Default to data_storage next to the app
            data_dir = base_path / "data_storage"
    
    # Create data directory if it doesn't exist
    if not data_dir.exists():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            _logger.info(f"Created data directory: {data_dir}")
        except OSError as e:
            _logger.warning(f"Could not create data directory: {e}")
    
    # Models and logs directories
    models_dir = data_dir / "models"
    logs_dir = data_dir / "logs"
    
    for dir_path in [models_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find TOTO data file (check multiple locations)
    toto_search_paths = [data_dir, base_path, base_path.parent]
    toto_files = ["toto_full_history.csv", "ToTo.csv", "toto_history.csv"]
    
    toto_csv = data_dir / "toto_full_history.csv"  # Primary
    toto_alt = None
    
    # Find alternative file
    for filename in toto_files:
        found = find_data_file(filename, toto_search_paths)
        if found and found != toto_csv:
            toto_alt = found
            break
    
    if toto_alt is None:
        toto_alt = base_path / "ToTo.csv"  # Fallback
    
    # Get AI settings from environment
    ai_random_state = int(_get_env_value("AI_RANDOM_STATE", 42))
    ai_n_estimators = int(_get_env_value("AI_N_ESTIMATORS", 100))
    ai_max_depth = int(_get_env_value("AI_MAX_DEPTH", 4))
    
    # Debug mode
    debug_mode = _get_env_value("DEBUG", "false").lower() in ("true", "1", "yes")
    
    config = Config(
        project_root=base_path,
        data_storage_dir=data_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        
        # Data files
        toto_csv_file=toto_csv,
        toto_alt_csv_file=toto_alt,
        four_d_csv_file=data_dir / "4d_full_history.csv",
        
        # TOTO settings
        toto_max_number=49,
        toto_pick_count=6,
        toto_analysis_window=100,
        
        # 4D settings
        four_d_sample_size=300,
        four_d_analysis_window=50,
        
        # AI settings
        ai_random_state=ai_random_state,
        ai_feature_window=50,
        ai_test_split=0.2,
        ai_n_estimators=ai_n_estimators,
        ai_max_depth=ai_max_depth,
        
        # Performance
        cache_max_age_seconds=3600,
        csv_read_retries=3,
        csv_read_retry_delay_seconds=0.1,
        
        # Flags
        skip_file_validation=skip_file_validation,
        debug_mode=debug_mode,
    )
    
    # Validate and log warnings
    if not skip_file_validation:
        warnings = config.validate()
        for warning in warnings:
            _logger.warning(warning)
    
    return config


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get the cached singleton configuration.
    
    Returns:
        The global Config instance
    """
    return load_config()


def reload_config() -> Config:
    """
    Force reload configuration (clears cache).
    
    Returns:
        Fresh Config instance
    """
    get_config.cache_clear()
    return get_config()


# ============================================================================
# GLOBAL EXPORTS (for backwards compatibility)
# ============================================================================

try:
    cfg = load_config()
    
    # Core paths
    PROJECT_ROOT = cfg.project_root
    DATA_STORAGE_DIR = cfg.data_storage_dir
    DATA_DIR = cfg.data_storage_dir  # Alias
    MODELS_DIR = cfg.models_dir
    LOGS_DIR = cfg.logs_dir
    
    # Data files
    TOTO_CSV_FILE = cfg.toto_csv_file
    TOTO_ALT_CSV_FILE = cfg.toto_alt_csv_file
    FOUR_D_CSV_FILE = cfg.four_d_csv_file
    
    # TOTO settings
    TOTO_MAX_NUMBER = cfg.toto_max_number
    TOTO_PICK_COUNT = cfg.toto_pick_count
    TOTO_ANALYSIS_WINDOW = cfg.toto_analysis_window
    
    # 4D settings
    FOUR_D_SAMPLE_SIZE = cfg.four_d_sample_size
    FOUR_D_ANALYSIS_WINDOW = cfg.four_d_analysis_window
    ANALYSIS_WINDOW = cfg.four_d_analysis_window  # Alias
    
    # AI settings
    AI_RANDOM_STATE = cfg.ai_random_state
    AI_FEATURE_WINDOW = cfg.ai_feature_window
    AI_TEST_SPLIT = cfg.ai_test_split
    AI_N_ESTIMATORS = cfg.ai_n_estimators
    AI_MAX_DEPTH = cfg.ai_max_depth
    
    # Performance settings
    CACHE_MAX_AGE_SECONDS = cfg.cache_max_age_seconds
    CSV_READ_RETRIES = cfg.csv_read_retries
    CSV_READ_RETRY_DELAY_SECONDS = cfg.csv_read_retry_delay_seconds
    
    # Flags
    DEBUG_MODE = cfg.debug_mode

except Exception as e:
    _logger.error(f"Configuration Error: {e}")
    # Set minimal defaults to prevent import errors
    PROJECT_ROOT = Path(".")
    DATA_STORAGE_DIR = Path("./data_storage")
    DATA_DIR = DATA_STORAGE_DIR
    TOTO_CSV_FILE = DATA_STORAGE_DIR / "toto_full_history.csv"
    FOUR_D_CSV_FILE = DATA_STORAGE_DIR / "4d_full_history.csv"


# ============================================================================
# TEST SECTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SG Lottery Suite - Configuration Test")
    print("=" * 60)
    
    config = get_config()
    
    print("\n[PATHS]")
    print(f"  Project Root: {config.project_root}")
    print(f"  Data Storage: {config.data_storage_dir}")
    print(f"  Models Dir: {config.models_dir}")
    
    print("\n[DATA FILES]")
    print(f"  TOTO CSV: {config.toto_csv_file}")
    print(f"  TOTO Alt: {config.toto_alt_csv_file}")
    print(f"  4D CSV: {config.four_d_csv_file}")
    
    # Check file existence
    toto_file = config.get_toto_data_file()
    if toto_file:
        print(f"  [OK] Using TOTO data: {toto_file.name}")
    else:
        print("  [!] No TOTO data file found")
    
    print("\n[AI SETTINGS]")
    print(f"  Random State: {config.ai_random_state}")
    print(f"  N Estimators: {config.ai_n_estimators}")
    print(f"  Max Depth: {config.ai_max_depth}")
    print(f"  Test Split: {config.ai_test_split}")
    
    print("\n[VALIDATION]")
    warnings = config.validate()
    if warnings:
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("  [OK] All checks passed")
    
    print("\n" + "=" * 60)
    print("Configuration Test Complete!")
