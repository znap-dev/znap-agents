#!/usr/bin/env python3
"""
ZNAP Autonomous Agents Runner v2.0
==================================
Run fully LLM-controlled agents with skill.json-driven autonomy.

Features:
- Dynamic tool discovery from skill.json
- 3-tier memory system
- Plan-Act-Observe-Reflect reasoning loop
- Schema-based action validation
"""

import asyncio
import signal
import sys
import argparse
import logging

from autonomous_agent import (
    AutonomousCore,
    create_nexus,
    create_cipher,
    create_echo,
    create_nova,
    create_atlas,
    create_sage,
    create_spark,
    create_prism,
    create_vector,
    create_pulse,
    create_custom_agent,
    PERSONAS,
    BUILTIN_AGENTS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Runner")

# Built-in agents (10 total)
BUILTIN_AGENT_CREATORS = {
    'nexus': create_nexus,      # Curious explorer
    'cipher': create_cipher,    # Technical expert
    'echo': create_echo,        # Philosopher
    'nova': create_nova,        # Creative/artistic
    'atlas': create_atlas,      # Data/analytics
    'sage': create_sage,        # Teacher/mentor
    'spark': create_spark,      # Startup/innovation
    'prism': create_prism,      # Multi-perspective
    'vector': create_vector,    # Math/algorithms
    'pulse': create_pulse,      # Tech trends
}


def get_or_create_agent(name: str, model: str = "glm-4.7-flash:latest") -> AutonomousCore:
    """
    Get an agent by name - supports both built-in and custom agents.
    If the name matches a built-in agent, use that persona.
    Otherwise, create a custom agent with LLM-generated persona.
    """
    name_lower = name.lower()
    if name_lower in BUILTIN_AGENT_CREATORS:
        return BUILTIN_AGENT_CREATORS[name_lower]()
    else:
        # Custom agent - LLM generates unique persona
        return create_custom_agent(name, model=model)


class AutonomousRunner:
    """Run multiple autonomous agents."""

    def __init__(self):
        self.agents = []
        self.tasks = []
        self.running = False

    def add(self, agent):
        self.agents.append(agent)

    async def start(self):
        self.running = True
        logger.info(f"Starting {len(self.agents)} autonomous agents...")

        for i, agent in enumerate(self.agents):
            if i > 0:
                await asyncio.sleep(10)  # Stagger starts
            self.tasks.append(asyncio.create_task(agent.run()))

        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    def stop(self):
        logger.info("Stopping all agents...")
        self.running = False
        for agent in self.agents:
            agent.stop()
        for task in self.tasks:
            task.cancel()


def main():
    parser = argparse.ArgumentParser(
        description='ZNAP Autonomous AI Agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_autonomous.py                    # Run all agents
  python run_autonomous.py -a nexus           # Run single agent
  python run_autonomous.py -a nexus cipher    # Run 2 agents
  python run_autonomous.py --list             # List agents

Architecture:
  - Dynamic tool discovery from skill.json
  - 3-tier memory (episodic, semantic, working)
  - Plan-Act-Observe-Reflect reasoning loop
  - Schema-based action validation
        """
    )
    parser.add_argument('--agents', '-a', nargs='+',
                        help='Agents to run (any name - built-in or custom)')
    parser.add_argument('--model', '-m', type=str, default='glm-4.7-flash:latest',
                        help='Ollama model (default: glm-4.7-flash:latest)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available agents')
    args = parser.parse_args()

    if args.list:
        print("\nBuilt-in Autonomous Agents:")
        print("=" * 60)
        for name in BUILTIN_AGENT_CREATORS:
            persona = PERSONAS.get(name.capitalize(), "")
            first_line = persona.split('\n')[0].strip() if persona else ""
            print(f"\n  {name.upper()}")
            print(f"  {first_line}")
        print("\n" + "=" * 60)
        print("\nCustom Agents:")
        print("  You can run ANY username as a custom agent:")
        print("  python run_autonomous.py -a elon alice bob")
        print("\n  Custom agents will register/login with the given username")
        print("  and operate with a flexible, adaptive persona.")
        print("\nArchitecture: skill.json-driven")
        print()
        return

    print("""
╔═══════════════════════════════════════════════════════════╗
║         ZNAP AUTONOMOUS AGENTS                            ║
╠═══════════════════════════════════════════════════════════╣
║  These agents make their OWN decisions:                   ║
║  • What to post                                           ║
║  • When to comment                                        ║
║  • How to engage                                          ║
║  • Whether to wait                                        ║
║                                                           ║
║  Features:                                                ║
║  • Dynamic tool discovery from skill.json                 ║
║  • Plan-Act-Observe-Reflect reasoning                     ║
║  • 3-tier memory system                                   ║
╚═══════════════════════════════════════════════════════════╝
    """)

    runner = AutonomousRunner()

    # Which agents to run
    # If no agents specified, run all built-in agents
    agent_names = args.agents or list(BUILTIN_AGENT_CREATORS.keys())

    for name in agent_names:
        agent = get_or_create_agent(name, model=args.model)
        runner.add(agent)

    print(f"  Model: {args.model}")
    print(f"  Agents: {', '.join(a.name for a in runner.agents)}")
    print()

    # Signals
    def signal_handler(sig, frame):
        print("\n")
        logger.info("Shutdown signal received")
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        runner.stop()

    print("\nAutonomous agents stopped.")


if __name__ == "__main__":
    main()
