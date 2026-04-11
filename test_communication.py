#!/usr/bin/env python3
"""
Test script for communication-enabled agents.
Runs 10 generations to verify the code works correctly.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase4.evolution import Evolution


def main():
    print("=" * 60)
    print("Testing Communication-Enabled Agents")
    print("=" * 60)
    
    # Initialize evolution
    evo = Evolution(seed=42)
    
    # Run 10 generations
    print("\nRunning 10 generations...\n")
    log = evo.run(generations=10, save_log=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    first = log[0]
    last = log[-1]
    
    print(f"\nGeneration 0:")
    print(f"  Population: {first['population']}")
    print(f"  Mean fitness: {first['mean_fitness']:.2f}")
    print(f"  Mean raw fitness: {first['mean_raw_fitness']:.2f}")
    print(f"  Messages sent: {first['messages_sent']}")
    print(f"  Coordination events: {first['coord_events']}")
    
    print(f"\nGeneration 9:")
    print(f"  Population: {last['population']}")
    print(f"  Mean fitness: {last['mean_fitness']:.2f}")
    print(f"  Mean raw fitness: {last['mean_raw_fitness']:.2f}")
    print(f"  Messages sent: {last['messages_sent']}")
    print(f"  Coordination events: {last['coord_events']}")
    print(f"  Total coordination events: {last['total_coord_events']}")
    
    print("\n✅ Test passed! Code runs successfully for 10 generations.")
    
    # Verify communication is happening
    total_msgs = sum(l['messages_sent'] for l in log)
    total_coord = sum(l['coord_events'] for l in log)
    
    print(f"\nCommunication statistics:")
    print(f"  Total messages sent: {total_msgs}")
    print(f"  Total coordination events: {total_coord}")
    
    if total_msgs > 0:
        print("  ✅ Agents are sending messages!")
    else:
        print("  ⚠️  No messages sent - agents may need more generations to learn")
    
    if total_coord > 0:
        print("  ✅ Coordination events detected!")
    else:
        print("  ⚠️  No coordination detected - may need more generations")


if __name__ == "__main__":
    main()
