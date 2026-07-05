"""
Preference score normalization tools.

Supports multiple normalization methods to map preference scores to the [0, 1] range.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import json


class PreferenceNormalizer:
    """Preference score normalizer"""
    
    SUPPORTED_METHODS = ['minmax', 'robust', 'sigmoid', 'quantile']
    
    def __init__(self, method: str = 'robust', **kwargs):
        """
        Initialize the normalizer.
        
        Args:
            method: Normalization method
                - 'minmax': Min-Max normalization (uses global min and max)
                - 'robust': Robust normalization (uses percentiles, insensitive to outliers)
                - 'sigmoid': Sigmoid normalization (suitable for standard normal distribution)
                - 'quantile': Quantile normalization (maps to percentiles)
            **kwargs: Method-specific parameters
                - percentile_range: (float, float) - percentile range for the robust method, default (5, 95)
                - sigmoid_scale: float - scaling factor for the sigmoid method, default 1.0
                - n_quantiles: int - number of quantiles for the quantile method, default 1000
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported normalization method: {method}. Available methods: {self.SUPPORTED_METHODS}")
        
        self.method = method
        self.stats = {}  # Store normalization statistics
        
        # Method-specific parameters
        self.percentile_range = kwargs.get('percentile_range', (5, 95))
        self.sigmoid_scale = kwargs.get('sigmoid_scale', 1.0)
        self.n_quantiles = kwargs.get('n_quantiles', 1000)
        
    def fit(self, scores: List[float], preference_name: str):
        """
        Fit the normalizer on the data and calculate the required statistics.
        
        Args:
            scores: List of raw scores
            preference_name: Name of the preference dimension
        """
        scores = np.array(scores, dtype=np.float64)
        
        if preference_name not in self.stats:
            self.stats[preference_name] = {}
        
        # Calculate basic statistics (required by all methods)
        self.stats[preference_name]['mean'] = float(np.mean(scores))
        self.stats[preference_name]['std'] = float(np.std(scores))
        self.stats[preference_name]['min'] = float(np.min(scores))
        self.stats[preference_name]['max'] = float(np.max(scores))
        
        # Method-specific statistics
        if self.method == 'minmax':
            # Min-Max: Only min and max are needed
            pass
            
        elif self.method == 'robust':
            # Robust: Use percentiles
            lower_p, upper_p = self.percentile_range
            self.stats[preference_name]['percentile_lower'] = float(np.percentile(scores, lower_p))
            self.stats[preference_name]['percentile_upper'] = float(np.percentile(scores, upper_p))
            
        elif self.method == 'sigmoid':
            # Sigmoid: Optionally use mean and std for standardization
            # Can be used directly for data that is already standardized (mean 0, std 1)
            pass
            
        elif self.method == 'quantile':
            # Quantile: Store quantile mappings
            self.stats[preference_name]['quantiles'] = np.linspace(
                np.min(scores), np.max(scores), self.n_quantiles
            ).tolist()
    
    def normalize(self, score: float, preference_name: str) -> float:
        """
        Normalize a single score to the [0, 1] range.
        
        Args:
            score: Raw score
            preference_name: Name of the preference dimension
            
        Returns:
            Normalized score (between [0, 1])
        """
        if preference_name not in self.stats:
            raise ValueError(f"Not fitted for preference dimension '{preference_name}'. Call fit() first.")
        
        stats = self.stats[preference_name]
        
        if self.method == 'minmax':
            # Min-Max normalization: (x - min) / (max - min)
            min_val, max_val = stats['min'], stats['max']
            if max_val == min_val:
                return 0.5
            normalized = (score - min_val) / (max_val - min_val)
            
        elif self.method == 'robust':
            # Robust normalization: Use percentiles instead of min/max
            lower_p = stats['percentile_lower']
            upper_p = stats['percentile_upper']
            if upper_p == lower_p:
                return 0.5
            normalized = (score - lower_p) / (upper_p - lower_p)
            
        elif self.method == 'sigmoid':
            # Sigmoid normalization: 1 / (1 + exp(-x * scale))
            # For a standard normal distribution, most values map to (0.1, 0.9)
            normalized = 1.0 / (1.0 + np.exp(-score * self.sigmoid_scale))
            
        elif self.method == 'quantile':
            # Quantile normalization: Map score to its percentile
            quantiles = np.array(stats['quantiles'])
            # Find the position of the score in the quantiles array
            position = np.searchsorted(quantiles, score)
            normalized = position / len(quantiles)
        
        # Ensure it falls within the [0, 1] range (minmax and robust might exceed this)
        return np.clip(normalized, 0.0, 1.0)
    
    def normalize_batch(self, scores: List[float], preference_name: str) -> List[float]:
        """
        Batch normalize scores.
        
        Args:
            scores: List of raw scores
            preference_name: Name of the preference dimension
            
        Returns:
            List of normalized scores
        """
        return [self.normalize(score, preference_name) for score in scores]
    
    def denormalize(self, normalized_score: float, preference_name: str) -> float:
        """
        Denormalize (restore [0, 1] values back to original range).
        
        Note: For sigmoid and quantile methods, denormalization might not be precise.
        
        Args:
            normalized_score: Normalized score (between [0, 1])
            preference_name: Name of the preference dimension
            
        Returns:
            Score in its original range
        """
        if preference_name not in self.stats:
            raise ValueError(f"Not fitted for preference dimension '{preference_name}'. Call fit() first.")
        
        stats = self.stats[preference_name]
        normalized_score = np.clip(normalized_score, 0.0, 1.0)
        
        if self.method == 'minmax':
            # Reverse Min-Max: x = normalized * (max - min) + min
            min_val, max_val = stats['min'], stats['max']
            return normalized_score * (max_val - min_val) + min_val
            
        elif self.method == 'robust':
            # Reverse robust normalization
            lower_p = stats['percentile_lower']
            upper_p = stats['percentile_upper']
            return normalized_score * (upper_p - lower_p) + lower_p
            
        elif self.method == 'sigmoid':
            # Reverse Sigmoid: x = -ln(1/y - 1) / scale
            # Prevent division by zero
            normalized_score = np.clip(normalized_score, 1e-7, 1 - 1e-7)
            return -np.log(1.0 / normalized_score - 1.0) / self.sigmoid_scale
            
        elif self.method == 'quantile':
            # Reverse quantile normalization
            quantiles = np.array(stats['quantiles'])
            index = int(normalized_score * (len(quantiles) - 1))
            return float(quantiles[index])
    
    def save(self, filepath: str):
        """
        Save the configuration and statistics of the normalizer.
        
        Args:
            filepath: Save path (JSON format)
        """
        config = {
            'method': self.method,
            'stats': self.stats,
            'params': {
                'percentile_range': self.percentile_range,
                'sigmoid_scale': self.sigmoid_scale,
                'n_quantiles': self.n_quantiles,
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'PreferenceNormalizer':
        """
        Load normalizer from a file.
        
        Args:
            filepath: Saved configuration file path
            
        Returns:
            Loaded normalizer instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Create normalizer instance
        normalizer = cls(
            method=config['method'],
            percentile_range=tuple(config['params']['percentile_range']),
            sigmoid_scale=config['params']['sigmoid_scale'],
            n_quantiles=config['params']['n_quantiles'],
        )
        normalizer.stats = config['stats']
        
        return normalizer
    
    def get_stats_summary(self) -> Dict:
        """
        Get the summary of normalization statistics.
        
        Returns:
            Dictionary containing statistics for all preference dimensions
        """
        summary = {
            'method': self.method,
            'params': {
                'percentile_range': self.percentile_range if self.method == 'robust' else None,
                'sigmoid_scale': self.sigmoid_scale if self.method == 'sigmoid' else None,
                'n_quantiles': self.n_quantiles if self.method == 'quantile' else None,
            },
            'stats': self.stats
        }
        return summary


def create_normalizer(
    method: str,
    scores_dict: Dict[str, List[float]],
    **kwargs
) -> PreferenceNormalizer:
    """
    Convenience function: Create and fit a normalizer.
    
    Args:
        method: Normalization method
        scores_dict: Dictionary {preference_name: list_of_scores}
        **kwargs: Method-specific parameters
        
    Returns:
        Fitted normalizer
        
    Example:
        >>> normalizer = create_normalizer(
        ...     method='robust',
        ...     scores_dict={
        ...         'harmless': [-2.5, -1.0, 0.0, 1.0, 2.5],
        ...         'helpful': [-1.5, -0.5, 0.5, 1.5, 3.0]
        ...     },
        ...     percentile_range=(10, 90)
        ... )
    """
    normalizer = PreferenceNormalizer(method=method, **kwargs)
    
    for preference_name, scores in scores_dict.items():
        normalizer.fit(scores, preference_name)
        print(f"Fitted normalizer for preference dimension '{preference_name}'")
        print(f"  Original Range: [{normalizer.stats[preference_name]['min']:.4f}, "
              f"{normalizer.stats[preference_name]['max']:.4f}]")
        print(f"  Mean: {normalizer.stats[preference_name]['mean']:.4f}, "
              f"Std: {normalizer.stats[preference_name]['std']:.4f}")
    
    return normalizer


if __name__ == "__main__":
    # Test example: using real HH-RLHF dataset
    import matplotlib.pyplot as plt
    from datasets import load_from_disk
    
    print("="*80)
    print("Normalization methods comparison test (using real HH-RLHF dataset)")
    print("="*80)
    
    # Load real dataset
    print("\nLoading dataset...")
    ds = load_from_disk('path/to/no_1-hh-rlhf')
    
    # Extract real harmless and helpful scores
    print("Extracting preference scores...")
    harmless_scores = np.array([float(item['harmless'].item()) for item in ds])
    helpful_scores = np.array([float(item['helpful'].item()) for item in ds])
    
    print(f"\nDataset size: {len(harmless_scores)}")
    print(f"Harmless - Min: {np.min(harmless_scores):.4f}, Max: {np.max(harmless_scores):.4f}, "
          f"Mean: {np.mean(harmless_scores):.4f}, Std: {np.std(harmless_scores):.4f}")
    print(f"Helpful  - Min: {np.min(helpful_scores):.4f}, Max: {np.max(helpful_scores):.4f}, "
          f"Mean: {np.mean(helpful_scores):.4f}, Std: {np.std(helpful_scores):.4f}")
    
    # Test different normalization methods
    methods_configs = [
        ('minmax', {}),
        ('robust', {'percentile_range': (5, 95)}),
        ('robust', {'percentile_range': (1, 99)}),  # Test wider percentiles
        ('sigmoid', {'sigmoid_scale': 1.0}),
    ]
    
    # Create subplots: show harmless and helpful normalized distributions for each method
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    
    for idx, (method, params) in enumerate(methods_configs):
        # Create normalizer
        param_str = f"({list(params.values())[0]})" if params else ""
        print(f"\n{'='*60}")
        print(f"Testing method: {method.upper()} {param_str}")
        print(f"{'='*60}")
        
        normalizer = create_normalizer(
            method=method,
            scores_dict={
                'harmless': harmless_scores.tolist(),
                'helpful': helpful_scores.tolist()
            },
            **params
        )
        
        # Normalize harmless
        harmless_norm = np.array(normalizer.normalize_batch(harmless_scores.tolist(), 'harmless'))
        
        # Normalize helpful
        helpful_norm = np.array(normalizer.normalize_batch(helpful_scores.tolist(), 'helpful'))
        
        # Plot harmless distribution
        ax_harmless = axes[0, idx]
        ax_harmless.hist(harmless_norm, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax_harmless.set_xlabel('Normalized Score', fontsize=10)
        ax_harmless.set_ylabel('Frequency', fontsize=10)
        title = f'{method.upper()}'
        if params:
            if 'percentile_range' in params:
                title += f'\n({params["percentile_range"][0]}-{params["percentile_range"][1]}%ile)'
            elif 'sigmoid_scale' in params:
                title += f'\n(scale={params["sigmoid_scale"]})'
        ax_harmless.set_title(f'Harmless - {title}', fontsize=11, fontweight='bold')
        ax_harmless.grid(axis='y', alpha=0.3)
        ax_harmless.axvline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='0.5')
        ax_harmless.legend(fontsize=8)
        ax_harmless.set_xlim(-0.05, 1.05)
        
        # Plot helpful distribution
        ax_helpful = axes[1, idx]
        ax_helpful.hist(helpful_norm, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        ax_helpful.set_xlabel('Normalized Score', fontsize=10)
        ax_helpful.set_ylabel('Frequency', fontsize=10)
        ax_helpful.set_title(f'Helpful - {title}', fontsize=11, fontweight='bold')
        ax_helpful.grid(axis='y', alpha=0.3)
        ax_helpful.axvline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='0.5')
        ax_helpful.legend(fontsize=8)
        ax_helpful.set_xlim(-0.05, 1.05)
        
        # Print detailed statistics
        print(f"\nHarmless normalized statistics:")
        print(f"  Min: {np.min(harmless_norm):.4f}, Max: {np.max(harmless_norm):.4f}")
        print(f"  Mean: {np.mean(harmless_norm):.4f}, Std: {np.std(harmless_norm):.4f}")
        print(f"  Q25: {np.percentile(harmless_norm, 25):.4f}, Median: {np.median(harmless_norm):.4f}, Q75: {np.percentile(harmless_norm, 75):.4f}")
        print(f"  Samples out of [0,1] range: {np.sum((harmless_norm < 0) | (harmless_norm > 1))}")
        
        print(f"\nHelpful normalized statistics:")
        print(f"  Min: {np.min(helpful_norm):.4f}, Max: {np.max(helpful_norm):.4f}")
        print(f"  Mean: {np.mean(helpful_norm):.4f}, Std: {np.std(helpful_norm):.4f}")
        print(f"  Q25: {np.percentile(helpful_norm, 25):.4f}, Median: {np.median(helpful_norm):.4f}, Q75: {np.percentile(helpful_norm, 75):.4f}")
        print(f"  Samples out of [0,1] range: {np.sum((helpful_norm < 0) | (helpful_norm > 1))}")
    
    plt.tight_layout()
    save_path = 'path/to/normalization_comparison_real_data.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Comparison plot saved to: {save_path}")
    print(f"{'='*80}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("Normalization Method Selection Recommendations:")
    print(f"{'='*80}")
    print("\n1. **MinMax**: ")
    print("   - Pros: Strictly guarantees output in [0,1] range, simple and intuitive.")
    print("   - Cons: Sensitive to outliers, might squeeze most data into the middle.")
    print("   - Recommended: Use when data quality is good and has no obvious outliers.")
    
    print("\n2. **Robust (5-95 percentiles)**:")
    print("   - Pros: Robust to outliers, preserves main data distribution characteristics.")
    print("   - Cons: A few samples might exceed [0,1] range (will be clipped).")
    print("   - Recommended: ⭐⭐⭐ Most recommended! Suitable for your standard normal distribution data.")
    
    print("\n3. **Robust (1-99 percentiles)**:")
    print("   - Pros: More conservative than 5-95, covers more data.")
    print("   - Cons: Still somewhat sensitive to extreme outliers.")
    print("   - Recommended: As an alternative to the Robust method.")
    
    print("\n4. **Sigmoid**:")
    print("   - Pros: Naturally adapts to standard normal distribution, smooth mapping.")
    print("   - Cons: Output distribution skews towards median, might lose resolution.")
    print("   - Recommended: Try it if you want to emphasize moderate preference values.")
    
    print(f"\n{'='*80}")
    print("Final Recommendation: Use **robust** method with percentile_range=(5, 95)")
    print("Reasoning: Your data is already a standard normal distribution. The Robust method can effectively handle long tails,")
    print("           while maintaining a good distribution for the majority of data, unaffected by extreme values unlike MinMax.")
    print(f"{'='*80}")
