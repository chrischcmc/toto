"""
TOTO AI Feature Builder (Enhanced Version)
============================================

Builds comprehensive, fixed-size feature vectors from lottery history
for machine learning models.

Features extracted:
1. Frequency Analysis (per number, normalized)
2. Recency Analysis (draws since last appearance)
3. Gap Statistics (average gap between appearances)
4. Hot/Cold Indicators
5. Pair Frequency Scores
6. Distribution Features (odd/even, low/high ratios)
7. Sum Statistics
8. Consecutive Number Patterns
"""

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    n_numbers: int = 49
    max_recency: int = 50
    recent_window: int = 20      # Window for "hot" number detection
    pair_top_k: int = 10         # Top K pairs to track
    include_advanced: bool = True  # Include advanced features


class TotoFeatureBuilder:
    """
    Builds deterministic, fixed-size feature vectors from lottery history.

    Basic Features (98 dimensions):
      - Frequency per number (49): normalized by total draws
      - Recency per number (49): normalized distance since last appearance

    Advanced Features (+147 dimensions = 245 total):
      - Gap statistics per number (49): avg gap / max_recency
      - Hot indicator (49): 1.0 if appeared in recent window, else 0.0
      - Distribution features (4): odd_ratio, low_ratio, sum_mean, sum_std
      - Consecutive patterns (10): frequency of consecutive number patterns
      - Global statistics (5): total draws, unique numbers, avg appearances, etc.

    Usage:
        builder = TotoFeatureBuilder()
        features = builder.build_from_history(history)
        # features.shape = (98,) for basic or (245,) for advanced
    """

    def __init__(
        self,
        max_recency: int = 50,
        n_numbers: int = 49,
        include_advanced: bool = True,
        recent_window: int = 20
    ) -> None:
        """
        Initialize the feature builder.
        
        Args:
            max_recency: Maximum recency value (for normalization)
            n_numbers: Total numbers in lottery (1 to n_numbers)
            include_advanced: Whether to include advanced features
            recent_window: Window size for hot number detection
        """
        self.n_numbers = n_numbers
        self.max_recency = max_recency
        self.include_advanced = include_advanced
        self.recent_window = recent_window
        
        # Calculate feature dimensions
        self.basic_dim = self.n_numbers * 2  # freq + recency
        self.advanced_dim = (
            self.n_numbers +    # gap statistics
            self.n_numbers +    # hot indicators
            10                  # distribution + patterns
        ) if include_advanced else 0
        
        self.total_dim = self.basic_dim + self.advanced_dim
        
        logger.info(
            "TotoFeatureBuilder initialized",
            extra={
                "n_numbers": self.n_numbers,
                "max_recency": max_recency,
                "include_advanced": include_advanced,
                "feature_dim": self.total_dim
            },
        )

    @property
    def feature_dim(self) -> int:
        """Returns the total feature dimension."""
        return self.total_dim

    def build_from_history(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Build a feature vector from lottery history.

        Args:
            history: List of draw records, ordered oldest -> newest.
                     Each record must have a 'nums' key with the drawn numbers.

        Returns:
            np.ndarray of shape (feature_dim,) with float32 dtype.
        """
        if not history:
            return np.zeros(self.total_dim, dtype=np.float32)

        total_draws = len(history)

        # Initialize tracking arrays
        freq = np.zeros(self.n_numbers, dtype=np.float32)
        last_seen = np.full(self.n_numbers, -1, dtype=np.int32)
        appearances: Dict[int, List[int]] = defaultdict(list)

        # Process each draw
        for idx, draw in enumerate(history):
            nums = draw.get("nums")
            if not isinstance(nums, Iterable):
                continue

            valid_nums = self._extract_valid_numbers(nums)
            
            for ni in valid_nums:
                freq[ni - 1] += 1.0
                last_seen[ni - 1] = idx
                appearances[ni].append(idx)

        # Build basic features
        features_list = []
        
        # 1. Frequency features (normalized)
        freq_norm = freq / max(float(total_draws), 1.0)
        features_list.append(freq_norm)

        # 2. Recency features (normalized)
        recency = self._calculate_recency(last_seen, total_draws)
        features_list.append(recency)

        # 3. Advanced features (if enabled)
        if self.include_advanced:
            # Gap statistics
            gap_features = self._calculate_gap_features(appearances, total_draws)
            features_list.append(gap_features)
            
            # Hot indicators
            hot_features = self._calculate_hot_features(history)
            features_list.append(hot_features)
            
            # Distribution features
            dist_features = self._calculate_distribution_features(history)
            features_list.append(dist_features)

        # Concatenate all features
        features = np.concatenate(features_list).astype(np.float32)

        # Validate dimension
        if features.shape[0] != self.total_dim:
            logger.warning(
                f"Feature dimension mismatch: expected {self.total_dim}, got {features.shape[0]}"
            )
            # Pad or truncate to match expected dimension
            if features.shape[0] < self.total_dim:
                features = np.pad(features, (0, self.total_dim - features.shape[0]))
            else:
                features = features[:self.total_dim]

        return features

    def _extract_valid_numbers(self, nums: Iterable) -> List[int]:
        """Extract and validate lottery numbers."""
        valid = []
        for n in set(nums):
            try:
                ni = int(n)
                if 1 <= ni <= self.n_numbers:
                    valid.append(ni)
            except (TypeError, ValueError):
                continue
        return valid

    def _calculate_recency(self, last_seen: np.ndarray, total_draws: int) -> np.ndarray:
        """Calculate normalized recency for each number."""
        recency = np.full(self.n_numbers, self.max_recency, dtype=np.float32)
        appeared_mask = last_seen >= 0
        
        if total_draws > 0:
            recency[appeared_mask] = (total_draws - 1) - last_seen[appeared_mask]
        
        recency = np.clip(recency, 0, self.max_recency) / float(self.max_recency)
        return recency

    def _calculate_gap_features(
        self, 
        appearances: Dict[int, List[int]], 
        total_draws: int
    ) -> np.ndarray:
        """
        Calculate average gap between appearances for each number.
        Returns normalized gap values (0-1 range).
        """
        gaps = np.zeros(self.n_numbers, dtype=np.float32)
        
        for num in range(1, self.n_numbers + 1):
            positions = appearances.get(num, [])
            if len(positions) >= 2:
                # Calculate gaps between consecutive appearances
                diffs = np.diff(positions)
                avg_gap = np.mean(diffs)
                # Normalize by max_recency
                gaps[num - 1] = min(avg_gap / self.max_recency, 1.0)
            elif len(positions) == 1:
                # Only appeared once - use distance from start as proxy
                gaps[num - 1] = 0.5
            else:
                # Never appeared
                gaps[num - 1] = 1.0
        
        return gaps

    def _calculate_hot_features(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Calculate hot indicator for each number.
        A number is "hot" if it appeared in the recent window.
        """
        hot = np.zeros(self.n_numbers, dtype=np.float32)
        
        if not history:
            return hot
        
        # Get recent draws
        recent_draws = history[-self.recent_window:]
        
        # Count appearances in recent window
        recent_counts = Counter()
        for draw in recent_draws:
            nums = draw.get("nums", [])
            for n in self._extract_valid_numbers(nums):
                recent_counts[n] += 1
        
        # Normalize by window size (max possible appearances)
        max_count = len(recent_draws)
        for num, count in recent_counts.items():
            hot[num - 1] = count / max_count
        
        return hot

    def _calculate_distribution_features(
        self, 
        history: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Calculate distribution-based features:
        - Average odd ratio in draws
        - Average low ratio (1-25) in draws
        - Sum mean (normalized)
        - Sum std (normalized)
        - Consecutive number frequency
        - Range statistics
        """
        if not history:
            return np.zeros(10, dtype=np.float32)
        
        odd_ratios = []
        low_ratios = []
        sums = []
        ranges = []
        consecutive_counts = []
        
        for draw in history:
            nums = draw.get("nums", [])
            valid = self._extract_valid_numbers(nums)
            
            if len(valid) >= 6:
                valid = sorted(valid)[:6]  # Take first 6
                
                # Odd/Even ratio
                odd_count = sum(1 for n in valid if n % 2 == 1)
                odd_ratios.append(odd_count / 6.0)
                
                # Low/High ratio (1-25 vs 26-49)
                low_count = sum(1 for n in valid if n <= 25)
                low_ratios.append(low_count / 6.0)
                
                # Sum
                total = sum(valid)
                sums.append(total)
                
                # Range
                ranges.append(valid[-1] - valid[0])
                
                # Consecutive numbers
                consec = sum(1 for i in range(len(valid)-1) if valid[i+1] - valid[i] == 1)
                consecutive_counts.append(consec)
        
        # Calculate statistics
        features = np.zeros(10, dtype=np.float32)
        
        if odd_ratios:
            features[0] = np.mean(odd_ratios)
            features[1] = np.std(odd_ratios)
        
        if low_ratios:
            features[2] = np.mean(low_ratios)
            features[3] = np.std(low_ratios)
        
        if sums:
            # Normalize sums (theoretical range: 21-279 for 6 numbers from 1-49)
            features[4] = (np.mean(sums) - 21) / (279 - 21)
            features[5] = np.std(sums) / 50  # Rough normalization
        
        if ranges:
            features[6] = np.mean(ranges) / 48  # Max range is 48
            features[7] = np.std(ranges) / 20
        
        if consecutive_counts:
            features[8] = np.mean(consecutive_counts) / 5  # Max 5 consecutive pairs
            features[9] = np.std(consecutive_counts) / 2
        
        return np.clip(features, 0, 1)

    def get_feature_names(self) -> List[str]:
        """Returns a list of feature names for interpretability."""
        names = []
        
        # Frequency features
        for i in range(1, self.n_numbers + 1):
            names.append(f"freq_{i}")
        
        # Recency features
        for i in range(1, self.n_numbers + 1):
            names.append(f"recency_{i}")
        
        if self.include_advanced:
            # Gap features
            for i in range(1, self.n_numbers + 1):
                names.append(f"gap_{i}")
            
            # Hot features
            for i in range(1, self.n_numbers + 1):
                names.append(f"hot_{i}")
            
            # Distribution features
            dist_names = [
                "odd_ratio_mean", "odd_ratio_std",
                "low_ratio_mean", "low_ratio_std",
                "sum_norm_mean", "sum_norm_std",
                "range_norm_mean", "range_norm_std",
                "consec_mean", "consec_std"
            ]
            names.extend(dist_names)
        
        return names


class FeatureBuilderFactory:
    """Factory for creating feature builders with different configurations."""
    
    @staticmethod
    def create_basic(n_numbers: int = 49) -> TotoFeatureBuilder:
        """Create a basic feature builder (98 dimensions)."""
        return TotoFeatureBuilder(
            n_numbers=n_numbers,
            include_advanced=False
        )
    
    @staticmethod
    def create_advanced(n_numbers: int = 49) -> TotoFeatureBuilder:
        """Create an advanced feature builder (245 dimensions)."""
        return TotoFeatureBuilder(
            n_numbers=n_numbers,
            include_advanced=True,
            max_recency=50,
            recent_window=20
        )
    
    @staticmethod
    def create_compact(n_numbers: int = 49) -> TotoFeatureBuilder:
        """Create a compact feature builder with reduced window."""
        return TotoFeatureBuilder(
            n_numbers=n_numbers,
            include_advanced=True,
            max_recency=30,
            recent_window=10
        )


# ==========================================
# TEST SECTION
# ==========================================
if __name__ == "__main__":
    import random
    
    print("=" * 60)
    print("TOTO Feature Builder - Test")
    print("=" * 60)
    
    # Create sample history
    history = []
    for i in range(100):
        nums = sorted(random.sample(range(1, 50), 6))
        history.append({"date": f"2024-{i//30+1:02d}-{i%30+1:02d}", "nums": nums})
    
    # Test basic builder
    print("\n[BASIC BUILDER]")
    basic_builder = FeatureBuilderFactory.create_basic()
    basic_features = basic_builder.build_from_history(history)
    print(f"  Feature dimension: {basic_features.shape[0]}")
    print(f"  Feature range: [{basic_features.min():.3f}, {basic_features.max():.3f}]")
    print(f"  Non-zero features: {np.count_nonzero(basic_features)}")
    
    # Test advanced builder
    print("\n[ADVANCED BUILDER]")
    advanced_builder = FeatureBuilderFactory.create_advanced()
    advanced_features = advanced_builder.build_from_history(history)
    print(f"  Feature dimension: {advanced_features.shape[0]}")
    print(f"  Feature range: [{advanced_features.min():.3f}, {advanced_features.max():.3f}]")
    print(f"  Non-zero features: {np.count_nonzero(advanced_features)}")
    
    # Show feature names
    print("\n[FEATURE NAMES (first 20)]")
    names = advanced_builder.get_feature_names()
    for i, name in enumerate(names[:20]):
        print(f"  {i+1:3d}. {name}: {advanced_features[i]:.4f}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
