"""
HPVD Engine Demo
================

Full demonstration of the Hybrid Probabilistic Vector Database:
1. Create synthetic trajectory data
2. Build HPVD indexes (sparse + dense)
3. Run similarity search
4. Show forecast with confidence intervals
5. Demonstrate quality metrics and abstention

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
    DistanceConfig
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
        
        # Generate embedding (PCA-like reduction)
        embedding = np.random.randn(256).astype(np.float32)
        embedding += (trend + volatility + structural) * 0.1  # Regime influence
        
        # Generate outcome labels based on regime (biased for demo)
        p_up = 0.5 + trend * 0.15 + volatility * 0.05
        label_h1 = 1 if np.random.random() < p_up else -1
        label_h5 = 1 if np.random.random() < p_up + 0.05 else -1
        
        traj = Trajectory(
            trajectory_id=f"traj_{i:04d}",
            asset_id=np.random.choice(assets),
            end_timestamp=base_date + timedelta(days=i % 365),
            matrix=matrix,
            embedding=embedding,
            label_h1=label_h1,
            label_h5=label_h5,
            return_h1=np.random.randn() * 0.02,
            return_h5=np.random.randn() * 0.05,
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
    """Run full HPVD demo"""
    
    print("=" * 70)
    print("🚀 HPVD ENGINE DEMO")
    print("   Hybrid Probabilistic Vector Database for Trajectory Intelligence")
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
    
    # ========== 4. Run Search ==========
    print("\n⚡ Running similarity search...")
    
    result = hpvd.search(query, k=25)
    
    print(f"\n📋 SEARCH RESULTS:")
    print("-" * 70)
    print(f"   K requested: {result.k_requested}")
    print(f"   K returned: {result.k_returned}")
    print(f"   Candidates after sparse filter: {result.candidates_after_sparse}")
    print(f"   Candidates after dense search: {result.candidates_after_dense}")
    
    # ========== 5. Show Latency Breakdown ==========
    print(f"\n⏱️  LATENCY BREAKDOWN:")
    print("-" * 70)
    for stage, ms in result.latency_breakdown.items():
        bar = "█" * int(ms / 2) + "░" * (20 - int(ms / 2))
        print(f"   {stage:20s} │{bar}│ {ms:.2f}ms")
    print(f"   {'TOTAL':20s} │{'█' * 20}│ {result.latency_ms:.2f}ms")
    
    # ========== 6. Show Top Analogs ==========
    print(f"\n🎯 TOP 10 ANALOGS:")
    print("-" * 70)
    print(f"   {'Rank':<5} {'ID':<12} {'Asset':<8} {'Distance':>10} {'Regime':>8} {'H1':>4} {'H5':>4}")
    print("-" * 70)
    
    for i, analog in enumerate(result.analogs[:10], 1):
        print(f"   {i:<5} {analog.trajectory_id:<12} {analog.asset_id:<8} "
              f"{analog.distance:>10.4f} {analog.regime_match:>8.2f} "
              f"{'+' if analog.label_h1 == 1 else '-':>4} "
              f"{'+' if analog.label_h5 == 1 else '-':>4}")
    
    # ========== 7. Show Forecast ==========
    print(f"\n📈 PROBABILISTIC FORECAST:")
    print("-" * 70)
    
    # H1 Forecast
    h1 = result.forecast_h1
    h1_bar = "█" * int(h1.p_up * 40) + "░" * (40 - int(h1.p_up * 40))
    print(f"   H1 (1-day):")
    print(f"      P(UP)   = {h1.p_up:.2%} [{h1.confidence_interval[0]:.2%} - {h1.confidence_interval[1]:.2%}]")
    print(f"      P(DOWN) = {h1.p_down:.2%}")
    print(f"      Entropy = {h1.entropy:.3f}")
    print(f"      [{h1_bar}]")
    
    # H5 Forecast
    h5 = result.forecast_h5
    h5_bar = "█" * int(h5.p_up * 40) + "░" * (40 - int(h5.p_up * 40))
    print(f"\n   H5 (5-day):")
    print(f"      P(UP)   = {h5.p_up:.2%} [{h5.confidence_interval[0]:.2%} - {h5.confidence_interval[1]:.2%}]")
    print(f"      P(DOWN) = {h5.p_down:.2%}")
    print(f"      Entropy = {h5.entropy:.3f}")
    print(f"      [{h5_bar}]")
    
    # ========== 8. Show Quality Metrics ==========
    print(f"\n✅ QUALITY METRICS:")
    print("-" * 70)
    
    aci_bar = "█" * int(result.aci * 20) + "░" * (20 - int(result.aci * 20))
    rc_bar = "█" * int(result.regime_coherence * 20) + "░" * (20 - int(result.regime_coherence * 20))
    
    print(f"   ACI (Analog Cohesion Index): {result.aci:.3f} [{aci_bar}]")
    print(f"   Regime Coherence:            {result.regime_coherence:.3f} [{rc_bar}]")
    
    # Quality gate evaluation
    print(f"\n   Quality Gates:")
    print(f"      ACI > 0.7?          {'✅ PASS' if result.aci > 0.7 else '⚠️  WARN'}")
    print(f"      Regime Coherence > 0.65? {'✅ PASS' if result.regime_coherence > 0.65 else '⚠️  WARN'}")
    print(f"      H1 Entropy < 0.9?   {'✅ PASS' if h1.entropy < 0.9 else '⚠️  WARN'}")
    
    # ========== 9. Abstention ==========
    print(f"\n🛑 ABSTENTION CHECK:")
    print("-" * 70)
    
    if result.should_abstain:
        print(f"   ⚠️  ABSTAIN: {result.abstention_reason}")
        print(f"   Recommendation: Do not trade based on this forecast")
    else:
        print(f"   ✅ CONFIDENT: No abstention needed")
        print(f"   Forecast can be used for decision making")
    
    # ========== 10. Summary ==========
    print(f"\n" + "=" * 70)
    print("📊 DEMO SUMMARY")
    print("=" * 70)
    print(f"   ✓ Trajectory database: {stats['total_trajectories']} trajectories")
    print(f"   ✓ Search latency: {result.latency_ms:.1f}ms (target: <50ms)")
    print(f"   ✓ Analogs found: {result.k_returned}")
    print(f"   ✓ H1 forecast: P(UP) = {h1.p_up:.1%}")
    print(f"   ✓ H5 forecast: P(UP) = {h5.p_up:.1%}")
    print(f"   ✓ Quality: ACI={result.aci:.2f}, RC={result.regime_coherence:.2f}")
    print("=" * 70)
    print("🎉 HPVD Engine Demo Complete!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    demo_hpvd_search()

