"""Quick script to inspect Scenario A data generation"""
from src.hpvd import SyntheticDataGenerator

gen = SyntheticDataGenerator(seed=42)
data = gen.generate_scenario_a()

print(f"Historical: {len(data['historical'])}")
print(f"Query: {len(data['query'])}")
print(f"\nFirst historical bundle:")
print(f"  Trajectory shape: {data['historical'][0].trajectory.shape}")
print(f"  DNA shape: {data['historical'][0].dna.shape}")
print(f"  Regime ID: {data['historical'][0].metadata.get('regime_id')}")
print(f"  Timestamp: {data['historical'][0].metadata.get('timestamp')}")
print(f"\nQuery bundle:")
print(f"  Trajectory shape: {data['query'][0].trajectory.shape}")
print(f"  DNA shape: {data['query'][0].dna.shape}")
print(f"  Regime ID: {data['query'][0].metadata.get('regime_id')}")
print(f"  Timestamp: {data['query'][0].metadata.get('timestamp')}")
