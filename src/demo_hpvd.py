"""
HPVD Engine Demo (Matrix22 – Retrieval Only)
============================================

Demonstration of the Hybrid Probabilistic Vector Database as a
pure retrieval + structural diagnostics engine:

1. Create synthetic trajectory data (with regimes)
2. Build HPVD indexes (sparse + dense)
3. Run similarity search
4. Show analogs and distance components

Important (Matrix22):
    - This demo is intentionally **outcome-blind**.
    - No probabilities, confidence intervals, entropy, or abstention
      decisions are computed here.

Usage:
    python -m src.demo_hpvd
"""

import numpy as np
from datetime import datetime, timedelta
import time

from src.hpvd import (
    Trajectory,
    HPVDEngine,
    HPVDConfig,
    DenseIndexConfig,
    DistanceConfig,
    HPVD_Output,
)


def create_synthetic_trajectories(n: int = 500) -> list:
    """
    Create synthetic trajectory data for demo
    
    Generates trajectories with different regime profiles
    """
    print(f"\n📊 Generating {n} synthetic trajectories...")
    
    trajectories = []
    assets = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "BTC-USD", "ETH-USD"]
    
    base_date = datetime(2020, 1, 1)
    
    for i in range(n):
        # Randomly assign regime
        trend = np.random.choice([-1, 0, 1])
        volatility = np.random.choice([-1, 0, 1])
        structural = np.random.choice([-1, 0, 1])
        
        # Generate matrix with regime-influenced patterns
        matrix = np.random.randn(60, 45).astype(np.float32)
        
        # Add regime-specific bias to make similar regimes cluster
        matrix[:, 0:8] += trend * 0.5  # Returns influenced by trend
        matrix[:, 8:18] += volatility * 0.3  # Volatility features
        matrix[:, 18:28] += structural * 0.2  # Structure features
        
        # Generate embedding (PCA-like reduction – synthetic only)
        embedding = np.random.randn(256).astype(np.float32)
        embedding += (trend + volatility + structural) * 0.1  # Regime influence

        traj = Trajectory(
            trajectory_id=f"traj_{i:04d}",
            asset_id=np.random.choice(assets),
            end_timestamp=base_date + timedelta(days=i % 365),
            matrix=matrix,
            embedding=embedding,
            trend_regime=trend,
            volatility_regime=volatility,
            structural_regime=structural,
            asset_class="crypto" if "USD" in np.random.choice(assets) else "equity"
        )
        trajectories.append(traj)
    
    print(f"   ✓ Created {len(trajectories)} trajectories")
    
    # Show distribution
    regimes = {}
    for t in trajectories:
        key = t.get_regime_tuple()
        regimes[key] = regimes.get(key, 0) + 1
    
    print(f"   ✓ Unique regime combinations: {len(regimes)}")
    
    return trajectories


def demo_hpvd_search():
    """Run HPVD retrieval demo (no forecasts/abstention)."""
    
    print("=" * 70)
    print("🚀 HPVD ENGINE DEMO (Matrix22 – Retrieval Only)")
    print("   Hybrid Probabilistic Vector Database for Evolutionary Analog Retrieval")
    print("=" * 70)
    
    # ========== 1. Create Data ==========
    trajectories = create_synthetic_trajectories(500)
    
    # ========== 2. Build HPVD ==========
    print("\n🔨 Building HPVD Engine...")
    
    config = HPVDConfig(
        default_k=25,
        search_k_multiplier=3,
        enable_sparse_filter=True,
        enable_reranking=True
    )
    
    hpvd = HPVDEngine(config)
    
    start_time = time.time()
    hpvd.build(trajectories)
    build_time = time.time() - start_time
    
    print(f"   ✓ Build time: {build_time:.2f}s")
    
    # Show statistics
    stats = hpvd.get_statistics()
    print(f"   ✓ Total trajectories: {stats['total_trajectories']}")
    print(f"   ✓ Unique regimes: {stats['sparse_index_stats']['unique_regimes']}")
    print(f"   ✓ Dense index vectors: {stats['dense_index_vectors']}")
    
    # ========== 3. Create Query ==========
    print("\n🔍 Creating query trajectory...")
    
    query = Trajectory(
        trajectory_id="query_001",
        asset_id="AAPL",
        end_timestamp=datetime.now(),
        matrix=np.random.randn(60, 45).astype(np.float32),
        embedding=np.random.randn(256).astype(np.float32),
        trend_regime=1,      # UP trend
        volatility_regime=0,  # MEDIUM volatility
        structural_regime=1,  # TREND following
        asset_class="equity"
    )
    
    print(f"   Query ID: {query.trajectory_id}")
    print(f"   Asset: {query.asset_id}")
    print(f"   Regime: trend={query.trend_regime}, vol={query.volatility_regime}, struct={query.structural_regime}")
    
    # ========== 4. Run Search (Matrix22: Family-based) ==========
    print("\n⚡ Running analog family search...")
    
    output = hpvd.search_families(query, max_candidates=100)
    
    print(f"\n📋 HPVD OUTPUT (Analog Families):")
    print("-" * 70)
    print(f"   Families formed: {len(output.analog_families)}")
    print(f"   Candidates considered: {output.retrieval_diagnostics['candidates_considered']}")
    print(f"   Candidates retrieved: {output.retrieval_diagnostics['candidates_retrieved']}")
    print(f"   Candidates admitted: {output.retrieval_diagnostics['candidates_admitted']}")
    print(f"   Candidates rejected: {output.retrieval_diagnostics['candidates_rejected']}")
    print(f"   Latency: {output.retrieval_diagnostics['latency_ms']:.1f}ms")
    
    # ========== 5. Show Analog Families ==========
    print(f"\n🎯 ANALOG FAMILIES:")
    print("-" * 70)
    
    for family in output.analog_families[:5]:  # Show top 5 families
        print(f"\n   Family: {family.family_id}")
        print(f"      Phase: {family.structural_signature.phase}")
        print(f"      Members: {family.coherence.size}")
        print(f"      Mean Confidence: {family.coherence.mean_confidence:.3f}")
        print(f"      Dispersion: {family.coherence.dispersion:.3f}")
        print(f"      Uncertainty Flags:")
        print(f"         phase_boundary: {family.uncertainty_flags.phase_boundary}")
        print(f"         weak_support: {family.uncertainty_flags.weak_support}")
        print(f"         partial_overlap: {family.uncertainty_flags.partial_overlap}")
        print(f"      Top 3 Members:")
        for i, member in enumerate(family.members[:3], 1):
            print(f"         {i}. {member.trajectory_id} (confidence: {member.confidence:.3f})")
    
    # ========== 6. Summary ==========
    print(f"\n" + "=" * 70)
    print("📊 DEMO SUMMARY")
    print("=" * 70)
    print(f"   ✓ Trajectory database: {stats['total_trajectories']} trajectories")
    print(f"   ✓ Search latency: {output.retrieval_diagnostics['latency_ms']:.1f}ms")
    print(f"   ✓ Analog families: {len(output.analog_families)}")
    print(f"   ✓ Total members: {sum(f.coherence.size for f in output.analog_families)}")
    print("=" * 70)
    print("🎉 HPVD Engine Demo Complete!")
    print("=" * 70)
    
    return output


if __name__ == "__main__":
    demo_hpvd_search()

