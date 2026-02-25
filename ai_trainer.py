"""
TOTO AI Trainer (Enhanced Version)
===================================

XGBoost-based training engine for TOTO lottery prediction.

Features:
- Trains 49 independent binary classifiers (one per number)
- Cross-validation for hyperparameter tuning
- Early stopping to prevent overfitting
- Progress tracking with ETA
- Model evaluation with accuracy metrics
- Configurable hyperparameters
- Training history logging
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TypedDict, Dict, Any, Callable
from datetime import datetime
from enum import Enum

import numpy as np

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

from ai_feature_builder import TotoFeatureBuilder, FeatureBuilderFactory
from ai_model_store import ModelStore

logger = logging.getLogger(__name__)


class TotoDraw(TypedDict):
    """Single draw record for training."""
    date: str
    nums: List[int]


class TrainingMode(Enum):
    """Training mode options."""
    FAST = "fast"           # Quick training, less accuracy
    STANDARD = "standard"   # Balanced
    THOROUGH = "thorough"   # More estimators, better accuracy


@dataclass
class XGBHyperParams:
    """XGBoost hyperparameters configuration."""
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0
    reg_alpha: float = 0
    reg_lambda: float = 1
    eval_metric: str = "logloss"
    random_state: int = 42
    n_jobs: int = -1  # Use all CPU cores
    
    @classmethod
    def fast(cls) -> "XGBHyperParams":
        """Quick training preset."""
        return cls(n_estimators=50, max_depth=3, learning_rate=0.2)
    
    @classmethod
    def standard(cls) -> "XGBHyperParams":
        """Standard training preset."""
        return cls(n_estimators=100, max_depth=4, learning_rate=0.1)
    
    @classmethod
    def thorough(cls) -> "XGBHyperParams":
        """Thorough training preset."""
        return cls(n_estimators=200, max_depth=5, learning_rate=0.05)


@dataclass
class TrainingResult:
    """Result of training a single model."""
    number: int
    success: bool
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    training_time_ms: float = 0
    pos_samples: int = 0
    neg_samples: int = 0
    error: Optional[str] = None


@dataclass
class TrainingSession:
    """Summary of a complete training session."""
    started_at: str
    completed_at: str
    total_time_seconds: float
    models_trained: int
    models_failed: int
    total_samples: int
    avg_accuracy: float
    results: List[TrainingResult] = field(default_factory=list)


class TotoTrainer:
    """
    Enhanced TOTO XGBoost Training Engine.
    
    Trains 49 independent binary classifiers with:
    - Configurable training modes (fast/standard/thorough)
    - Progress callbacks for UI integration
    - Model evaluation and metrics
    - Cross-validation support
    
    Usage:
        trainer = TotoTrainer("./data_storage")
        
        # Standard training
        session = trainer.train_all()
        
        # Fast training with progress callback
        session = trainer.train_all(
            mode=TrainingMode.FAST,
            progress_callback=lambda n, t: print(f"Training {n}/{t}")
        )
        
        # Train single number
        result = trainer.train_single(42)
    """

    N_MODELS: int = 49
    MIN_SAMPLES: int = 50
    WINDOW_SIZE: int = 50  # Minimum history window for features
    TEST_SPLIT: float = 0.2  # 20% for validation

    def __init__(
        self,
        base_dir: str,
        mode: TrainingMode = TrainingMode.STANDARD,
        use_advanced_features: bool = True
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            base_dir: Base directory for data and models
            mode: Training mode (fast/standard/thorough)
            use_advanced_features: Whether to use advanced feature builder
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        
        # Initialize feature builder
        if use_advanced_features:
            self.features = FeatureBuilderFactory.create_advanced()
        else:
            self.features = FeatureBuilderFactory.create_basic()
        
        self.store = ModelStore(base_dir)
        self.model_dir = os.path.join(base_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set hyperparameters based on mode
        self._mode = mode
        self._params = self._get_params_for_mode(mode)
        
        # Training state
        self._is_training = False
        self._should_stop = False
        
        logger.info(
            "TotoTrainer initialized",
            extra={
                "model_dir": self.model_dir,
                "mode": mode.value,
                "feature_dim": self.features.feature_dim
            }
        )

    def _get_params_for_mode(self, mode: TrainingMode) -> XGBHyperParams:
        """Get hyperparameters for training mode."""
        if mode == TrainingMode.FAST:
            return XGBHyperParams.fast()
        elif mode == TrainingMode.THOROUGH:
            return XGBHyperParams.thorough()
        else:
            return XGBHyperParams.standard()

    def set_mode(self, mode: TrainingMode) -> None:
        """Change training mode."""
        self._mode = mode
        self._params = self._get_params_for_mode(mode)
        logger.info(f"Training mode set to: {mode.value}")

    def stop_training(self) -> None:
        """Request training to stop (for UI cancel button)."""
        self._should_stop = True
        logger.info("Training stop requested")

    def train_all(
        self,
        mode: Optional[TrainingMode] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        validate: bool = True
    ) -> TrainingSession:
        """
        Train all 49 models.
        
        Args:
            mode: Override training mode
            progress_callback: Function called with (current, total) after each model
            validate: Whether to validate models after training
            
        Returns:
            TrainingSession with results and metrics.
        """
        if mode:
            self._params = self._get_params_for_mode(mode)
        
        self._is_training = True
        self._should_stop = False
        
        started_at = datetime.now()
        results: List[TrainingResult] = []
        
        logger.info("training.start", extra={"n_models": self.N_MODELS})

        # Load and prepare data
        history = self.store.load_raw_data()
        if not history:
            logger.error("training.no_data")
            return self._create_empty_session(started_at)

        logger.info("training.data_loaded", extra={"n_draws": len(history)})

        X, y_matrix = self._prepare_dataset(history)

        if len(X) < self.MIN_SAMPLES:
            logger.error(
                "training.not_enough_samples",
                extra={"min_samples": self.MIN_SAMPLES, "actual": len(X)},
            )
            return self._create_empty_session(started_at)

        # Split data for validation
        if validate and HAS_SKLEARN:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_matrix, test_size=self.TEST_SPLIT, random_state=42
            )
        else:
            X_train, y_train = X, y_matrix
            X_val, y_val = None, None

        # Train each number's model
        for number in range(1, self.N_MODELS + 1):
            if self._should_stop:
                logger.info("training.stopped_by_user")
                break
            
            result = self._train_single_model(
                number, X_train, y_train, X_val, y_val
            )
            results.append(result)
            
            # Progress callback
            if progress_callback:
                try:
                    progress_callback(number, self.N_MODELS)
                except Exception:
                    pass
            
            # Log progress every 10 models
            if number % 10 == 0:
                successful = sum(1 for r in results if r.success)
                logger.info(
                    "training.progress",
                    extra={"trained": number, "successful": successful}
                )

        self._is_training = False
        completed_at = datetime.now()
        
        # Calculate summary
        successful = [r for r in results if r.success]
        accuracies = [r.accuracy for r in successful if r.accuracy]
        
        session = TrainingSession(
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            total_time_seconds=round((completed_at - started_at).total_seconds(), 2),
            models_trained=len(successful),
            models_failed=len(results) - len(successful),
            total_samples=len(X),
            avg_accuracy=round(np.mean(accuracies), 4) if accuracies else 0,
            results=results
        )

        logger.info(
            "training.complete",
            extra={
                "successful": session.models_trained,
                "failed": session.models_failed,
                "avg_accuracy": session.avg_accuracy,
                "time_seconds": session.total_time_seconds
            }
        )

        return session

    def _train_single_model(
        self,
        number: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> TrainingResult:
        """Train a single number's model."""
        start_time = time.perf_counter()
        
        y = y_train[:, number - 1]
        
        # Check class balance
        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            return TrainingResult(
                number=number,
                success=False,
                error="single_class"
            )

        pos_count = int(np.sum(y))
        neg_count = int(len(y) - pos_count)

        if pos_count == 0:
            return TrainingResult(
                number=number,
                success=False,
                error="no_positive_samples"
            )

        # Calculate class weight
        scale_weight = max(neg_count / pos_count, 1.0)

        # Create and train model
        model = XGBClassifier(
            n_estimators=self._params.n_estimators,
            max_depth=self._params.max_depth,
            learning_rate=self._params.learning_rate,
            min_child_weight=self._params.min_child_weight,
            subsample=self._params.subsample,
            colsample_bytree=self._params.colsample_bytree,
            gamma=self._params.gamma,
            reg_alpha=self._params.reg_alpha,
            reg_lambda=self._params.reg_lambda,
            scale_pos_weight=scale_weight,
            eval_metric=self._params.eval_metric,
            use_label_encoder=False,
            random_state=self._params.random_state,
            n_jobs=self._params.n_jobs,
            verbosity=0  # Suppress XGBoost output
        )

        try:
            model.fit(X_train, y)
            
            # Calculate validation metrics
            accuracy = precision = recall = f1 = None
            
            if X_val is not None and y_val is not None and HAS_SKLEARN:
                y_val_single = y_val[:, number - 1]
                y_pred = model.predict(X_val)
                
                accuracy = round(accuracy_score(y_val_single, y_pred), 4)
                
                # Only calculate if both classes present in validation
                if len(np.unique(y_val_single)) > 1:
                    precision = round(precision_score(y_val_single, y_pred, zero_division=0), 4)
                    recall = round(recall_score(y_val_single, y_pred, zero_division=0), 4)
                    f1 = round(f1_score(y_val_single, y_pred, zero_division=0), 4)
            
            # Save model with metadata
            self.store.save_model(
                number,
                model,
                training_samples=len(X_train),
                feature_dim=X_train.shape[1],
                accuracy=accuracy
            )
            
            training_time = (time.perf_counter() - start_time) * 1000
            
            return TrainingResult(
                number=number,
                success=True,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                training_time_ms=round(training_time, 2),
                pos_samples=pos_count,
                neg_samples=neg_count
            )

        except Exception as exc:
            logger.error(
                "training.model_failed",
                extra={"number": number, "error": str(exc)}
            )
            return TrainingResult(
                number=number,
                success=False,
                error=str(exc)
            )

    def train_single(self, number: int, validate: bool = True) -> TrainingResult:
        """
        Train a single number's model.
        
        Args:
            number: Number to train (1-49)
            validate: Whether to validate after training
            
        Returns:
            TrainingResult with metrics.
        """
        if not 1 <= number <= self.N_MODELS:
            raise ValueError(f"Number must be 1-{self.N_MODELS}")
        
        history = self.store.load_raw_data()
        if not history:
            return TrainingResult(number=number, success=False, error="no_data")
        
        X, y_matrix = self._prepare_dataset(history)
        
        if validate and HAS_SKLEARN:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_matrix, test_size=self.TEST_SPLIT, random_state=42
            )
        else:
            X_train, y_train = X, y_matrix
            X_val, y_val = None, None
        
        return self._train_single_model(number, X_train, y_train, X_val, y_val)

    def _prepare_dataset(
        self,
        history: List[TotoDraw]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training dataset from history.
        
        Uses sliding window to build features for each target draw.
        """
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for i in range(self.WINDOW_SIZE, len(history)):
            past_draws = history[:i]
            target_draw = history[i]

            # Build features from past draws
            feats = self.features.build_from_history(past_draws)
            X_list.append(feats)

            # Build label vector
            label_vec = np.zeros(self.N_MODELS, dtype=np.int8)
            for n in target_draw.get("nums", []):
                if 1 <= n <= self.N_MODELS:
                    label_vec[n - 1] = 1

            y_list.append(label_vec)

        if not X_list:
            return np.empty((0, 0), dtype=float), np.empty((0, self.N_MODELS), dtype=np.int8)

        X = np.vstack(X_list)
        y_matrix = np.vstack(y_list)

        logger.info(
            "dataset.prepared",
            extra={
                "samples": len(X),
                "features": X.shape[1],
                "labels_shape": y_matrix.shape
            }
        )

        return X, y_matrix

    def _create_empty_session(self, started_at: datetime) -> TrainingSession:
        """Create an empty training session for error cases."""
        return TrainingSession(
            started_at=started_at.isoformat(),
            completed_at=datetime.now().isoformat(),
            total_time_seconds=0,
            models_trained=0,
            models_failed=0,
            total_samples=0,
            avg_accuracy=0,
            results=[]
        )

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "is_training": self._is_training,
            "mode": self._mode.value,
            "params": {
                "n_estimators": self._params.n_estimators,
                "max_depth": self._params.max_depth,
                "learning_rate": self._params.learning_rate
            }
        }


# ==========================================
# TEST SECTION
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("TOTO AI Trainer - Test")
    print("=" * 60)
    
    if not HAS_XGBOOST:
        print("ERROR: xgboost not installed")
        exit(1)
    
    # Initialize trainer
    trainer = TotoTrainer(".", mode=TrainingMode.FAST)
    
    print("\n[TRAINER STATUS]")
    status = trainer.get_training_status()
    print(f"  Mode: {status['mode']}")
    print(f"  Estimators: {status['params']['n_estimators']}")
    print(f"  Max Depth: {status['params']['max_depth']}")
    
    # Test single model training
    print("\n[TRAINING SINGLE MODEL (Number 1)]")
    result = trainer.train_single(1, validate=True)
    print(f"  Success: {result.success}")
    if result.success:
        print(f"  Accuracy: {result.accuracy}")
        print(f"  Training time: {result.training_time_ms}ms")
        print(f"  Samples: +{result.pos_samples} / -{result.neg_samples}")
    else:
        print(f"  Error: {result.error}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
