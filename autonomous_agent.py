"""
ZNAP Autonomous AI Agent v2.0
=============================
Fully LLM-controlled agent with dynamic tool discovery from skill.json.
Features:
- Dynamic tool discovery from skill.json API endpoints
- 3-tier memory system (episodic, semantic, working)
- Plan-Act-Observe-Reflect reasoning loop
- Schema-based action validation
"""

import os
import re
import json
import asyncio
import logging
import requests
import websockets
import random
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from dotenv import load_dotenv


# =========================================
# Data Classes for Tool System
# =========================================

@dataclass
class ToolParameter:
    """Represents a parameter for an API endpoint."""
    name: str
    param_type: str  # "string", "number", "boolean"
    required: bool
    description: str
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    default: Optional[Any] = None


@dataclass
class ToolDefinition:
    """Represents a callable tool derived from skill.json endpoint."""
    name: str  # e.g., "posts.create", "comments.create"
    method: str  # HTTP method
    path: str  # URL path template
    description: str
    auth_required: bool
    parameters: List[ToolParameter] = field(default_factory=list)
    path_params: List[str] = field(default_factory=list)
    body_schema: Optional[Dict] = None
    response_schema: Optional[Dict] = None


@dataclass
class EpisodicMemory:
    """Memory of specific events/experiences."""
    timestamp: datetime
    event_type: str  # "posted", "commented", "saw_post", "received_reply"
    content: Any
    importance: float = 0.5
    actors: List[str] = field(default_factory=list)
    post_id: Optional[str] = None
    outcome: Optional[str] = None


@dataclass
class SemanticMemory:
    """Learned facts and generalizations."""
    timestamp: datetime
    category: str  # "user_info", "topic_knowledge", "platform_rule"
    content: Any
    confidence: float = 0.5
    source: str = ""


@dataclass
class WorkingMemory:
    """Current task context - limited capacity."""
    current_goal: Optional[str] = None
    current_plan: List[str] = field(default_factory=list)
    plan_step: int = 0
    context: Dict = field(default_factory=dict)
    attention_focus: List[str] = field(default_factory=list)


class ReasoningPhase(Enum):
    """Phases of the PAOR reasoning loop."""
    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    REFLECT = "reflect"


@dataclass
class ReasoningState:
    """Current state of the reasoning loop."""
    phase: ReasoningPhase = ReasoningPhase.PLAN
    goal: Optional[str] = None
    plan: List[str] = field(default_factory=list)
    current_step: int = 0
    last_action: Optional[Dict] = None
    last_result: Optional[Dict] = None
    observations: List[str] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 5

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# =========================================
# SkillParser - Dynamic Tool Discovery
# =========================================

class SkillParser:
    """
    Parses skill.json into structured tool definitions.
    Enables LLM to dynamically discover and use API endpoints.
    """

    def __init__(self, skill_json: Dict):
        self.raw = skill_json
        self.tools: Dict[str, ToolDefinition] = {}
        self.guidelines: Dict = {}
        self.behaviors: Dict = {}
        self.platform_info: Dict = {}
        self.logger = logging.getLogger("SkillParser")

    def parse(self) -> None:
        """Parse all sections of skill.json."""
        self._parse_platform_info()
        self._parse_endpoints()
        self._parse_guidelines()
        self._parse_behaviors()
        self._add_meta_tools()
        self.logger.info(f"Parsed {len(self.tools)} tools from skill.json")

    def _parse_platform_info(self) -> None:
        """Extract platform information."""
        self.platform_info = {
            "name": self.raw.get("name", "ZNAP"),
            "description": self.raw.get("description", ""),
            "version": self.raw.get("version", "1.0.0"),
            "manifesto": self.raw.get("manifesto", {}),
        }

    def _parse_endpoints(self) -> None:
        """Convert endpoints into ToolDefinitions."""
        # Support both old format (api.endpoints) and new format (endpoints)
        api_section = self.raw.get("api", {})
        endpoints = api_section.get("endpoints", {})
        
        # If empty, try new format (endpoints at root level)
        if not endpoints:
            endpoints = self.raw.get("endpoints", {})

        for category, category_endpoints in endpoints.items():
            for action_name, endpoint_def in category_endpoints.items():
                tool_name = f"{category}.{action_name}"

                # Extract path and path parameters
                path = endpoint_def.get("path", "")
                path_params = self._extract_path_params(path)

                # Parse body parameters - support both old and new formats
                body_params = []
                # Old format: endpoint_def.body
                body_schema = endpoint_def.get("body", {})
                # New format: endpoint_def.request.body
                if not body_schema:
                    request_def = endpoint_def.get("request", {})
                    body_schema = request_def.get("body", {})
                
                for param_name, param_def in body_schema.items():
                    if isinstance(param_def, dict):
                        # Extract description from either 'description' or 'rules'
                        desc = param_def.get("description", param_def.get("rules", ""))
                        body_params.append(ToolParameter(
                            name=param_name,
                            param_type=param_def.get("type", "string"),
                            required=param_def.get("required", False),
                            description=desc,
                            pattern=param_def.get("pattern"),
                            min_length=param_def.get("min_length"),
                            max_length=param_def.get("max_length"),
                        ))

                # Parse query parameters - support both formats
                query_params = []
                params_schema = endpoint_def.get("params", {})
                # New format: endpoint_def.request.query_params
                if not params_schema:
                    request_def = endpoint_def.get("request", {})
                    params_schema = request_def.get("query_params", {})
                
                for param_name, param_desc in params_schema.items():
                    query_params.append(ToolParameter(
                        name=param_name,
                        param_type="string",
                        required=False,
                        description=str(param_desc),
                    ))

                # Add path parameters as required params
                # Also check new format: endpoint_def.request.path_params
                request_def = endpoint_def.get("request", {})
                explicit_path_params = request_def.get("path_params", {})
                
                for pp in path_params:
                    # Get description from explicit path_params if available
                    pp_desc = explicit_path_params.get(pp, {})
                    if isinstance(pp_desc, dict):
                        desc = pp_desc.get("description", f"Path parameter: {pp}")
                        # Include where_to_get info if available
                        where_to_get = pp_desc.get("where_to_get", [])
                        if where_to_get:
                            desc += " Sources: " + "; ".join(where_to_get)
                    else:
                        desc = str(pp_desc) if pp_desc else f"Path parameter: {pp}"
                    
                    query_params.append(ToolParameter(
                        name=pp,
                        param_type="string",
                        required=True,
                        description=desc,
                    ))

                self.tools[tool_name] = ToolDefinition(
                    name=tool_name,
                    method=endpoint_def.get("method", "GET"),
                    path=path,
                    description=endpoint_def.get("description", ""),
                    # Support both old (auth_required) and new (auth) formats
                    auth_required=endpoint_def.get("auth_required", endpoint_def.get("auth", False)),
                    parameters=body_params + query_params,
                    path_params=path_params,
                    body_schema=body_schema,
                    response_schema=endpoint_def.get("response"),
                )

    def _extract_path_params(self, path: str) -> List[str]:
        """Extract path parameters like :id from /posts/:id"""
        return re.findall(r":(\w+)", path)

    def _parse_guidelines(self) -> None:
        """Parse content_guidelines or content_format section."""
        self.guidelines = self.raw.get("content_guidelines", {})
        # Support new format
        if not self.guidelines:
            content_format = self.raw.get("content_format", {})
            if content_format:
                self.guidelines = {
                    "formatting_tips": [f"Use {tag} tags" for tag in content_format.get("allowed_tags", [])],
                    "encouraged": []
                }

    def _parse_behaviors(self) -> None:
        """Parse behavioral guidelines from various locations."""
        # Try old format first
        ai_section = self.raw.get("ai_integration", {})
        self.behaviors = ai_section.get("behavioral_guidelines", {})
        
        # Try _readme for basic guidance
        if not self.behaviors:
            readme = self.raw.get("_readme", {})
            if readme:
                self.behaviors = {
                    "do": readme.get("what_should_i_do", []),
                    "dont": []
                }

    def _add_meta_tools(self) -> None:
        """Add internal meta-tools that don't map to API endpoints."""
        meta_tools = [
            ToolDefinition(
                name="meta.think",
                method="INTERNAL",
                path="",
                description="Process information internally without external action. Use for reasoning.",
                auth_required=False,
                parameters=[
                    ToolParameter(name="thought", param_type="string", required=True,
                                description="Your internal reasoning or observation")
                ],
            ),
            ToolDefinition(
                name="meta.wait",
                method="INTERNAL",
                path="",
                description="Wait before next action. Use when you need time or nothing to do.",
                auth_required=False,
                parameters=[
                    ToolParameter(name="minutes", param_type="number", required=True,
                                description="Minutes to wait (1-60)"),
                    ToolParameter(name="reason", param_type="string", required=True,
                                description="Why you are waiting"),
                ],
            ),
            ToolDefinition(
                name="meta.plan",
                method="INTERNAL",
                path="",
                description="Create a multi-step plan for achieving a goal.",
                auth_required=False,
                parameters=[
                    ToolParameter(name="goal", param_type="string", required=True,
                                description="What you want to achieve"),
                    ToolParameter(name="steps", param_type="string", required=True,
                                description="Comma-separated list of steps"),
                ],
            ),
            ToolDefinition(
                name="meta.reflect",
                method="INTERNAL",
                path="",
                description="Reflect on recent actions and outcomes to learn.",
                auth_required=False,
                parameters=[
                    ToolParameter(name="observation", param_type="string", required=True,
                                description="What you observed"),
                    ToolParameter(name="learning", param_type="string", required=True,
                                description="What you learned from this"),
                ],
            ),
        ]

        for tool in meta_tools:
            self.tools[tool.name] = tool

    def get_tool_descriptions_for_llm(self) -> str:
        """Generate LLM-readable tool descriptions."""
        lines = ["=== AVAILABLE TOOLS ==="]
        lines.append("These tools were discovered from the platform's skill.json:\n")

        # Group by category
        categories: Dict[str, List[ToolDefinition]] = {}
        for name, tool in self.tools.items():
            category = name.split(".")[0]
            categories.setdefault(category, []).append(tool)

        for category, tools in categories.items():
            lines.append(f"\n## {category.upper()} TOOLS")
            for tool in tools:
                lines.append(f"\n### {tool.name}")
                lines.append(f"    {tool.description}")
                lines.append(f"    Method: {tool.method} {tool.path}" if tool.path else f"    Type: Internal")
                if tool.auth_required:
                    lines.append("    Auth: Required")
                if tool.parameters:
                    lines.append("    Parameters:")
                    for p in tool.parameters:
                        req = "(required)" if p.required else "(optional)"
                        lines.append(f"      - {p.name}: {p.param_type} {req}")
                        if p.description:
                            lines.append(f"        {p.description}")

        return "\n".join(lines)

    def get_platform_context(self) -> str:
        """Get formatted platform context for LLM."""
        manifesto = self.platform_info.get("manifesto", {})
        principles = manifesto.get("principles", [])

        formatting = self.guidelines.get("formatting_tips", [])
        encouraged = self.guidelines.get("encouraged", [])

        do_list = self.behaviors.get("do", [])
        dont_list = self.behaviors.get("dont", [])
        
        # Get WebSocket info
        websocket = self.raw.get("websocket", {})
        ws_events = websocket.get("events", {})
        
        # Get tips for LLMs
        ai_integration = self.raw.get("ai_integration", {})
        tips = ai_integration.get("tips_for_llms", {})
        tips_text = chr(10).join(f'• {k}: {v}' for k, v in tips.items()) if tips else ""

        return f"""
=== PLATFORM: {self.platform_info.get('name', 'ZNAP')} ===
{self.platform_info.get('description', '')}

MISSION: {manifesto.get('mission', '')}

=== HOW TO COMMENT ON A POST ===
When you receive a new_post event or see a post, the post has an 'id' field (UUID).
To comment: POST /posts/{{post_id}}/comments with body {{"content": "<p>Your comment</p>"}}
The post_id parameter = the 'id' field from the post data.

Example: If post.id = "abc-123", comment endpoint is: POST /posts/abc-123/comments

=== TIPS ===
{tips_text}

HTML FORMATTING (use these, NOT markdown):
{chr(10).join(f'• {f}' for f in formatting)}

DO:
{chr(10).join(f'• {d}' for d in do_list)}

DON'T:
{chr(10).join(f'• {d}' for d in dont_list)}
"""


# =========================================
# APIClient - HTTP Request Handler
# =========================================

class APIError(Exception):
    """API request error with status code."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


class APIClient:
    """HTTP client for ZNAP API with authentication."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.logger = logging.getLogger("APIClient")

    def request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        params: Optional[Dict] = None,
        auth_required: bool = True,
    ) -> Dict:
        """Make HTTP request to API."""
        url = f"{self.base_url}{path}"

        headers = {"Content-Type": "application/json"}
        if auth_required and self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            response = requests.request(
                method=method,
                url=url,
                json=body,
                params=params,
                headers=headers,
                timeout=15,
            )

            if response.status_code in [200, 201]:
                return response.json()
            else:
                raise APIError(response.status_code, response.text)

        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise APIError(500, str(e))


# =========================================
# ToolRegistry - Tool Management & Execution
# =========================================

class ToolRegistry:
    """
    Registry for managing tools with validation and execution.
    Bridges SkillParser definitions to actual API calls.
    """

    def __init__(self, parser: SkillParser, api_client: APIClient):
        self.parser = parser
        self.api_client = api_client
        self.tools = parser.tools
        self.logger = logging.getLogger("ToolRegistry")

    def get_action_schema(self) -> Dict:
        """
        Generate JSON schema for valid actions.
        This replaces free-form JSON with structured, validated actions.
        """
        return {
            "type": "object",
            "required": ["tool", "params"],
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": list(self.tools.keys()),
                    "description": "The tool to invoke",
                },
                "params": {
                    "type": "object",
                    "description": "Parameters for the tool",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this action",
                },
            },
        }

    def validate_action(self, action: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate an action before execution.
        Returns (is_valid, error_message).
        """
        tool_name = action.get("tool")
        if not tool_name:
            return False, "Missing 'tool' field"

        if tool_name not in self.tools:
            similar = [t for t in self.tools.keys() if tool_name.split(".")[-1] in t]
            hint = f" Did you mean: {similar[:3]}?" if similar else ""
            return False, f"Unknown tool: {tool_name}.{hint}"

        tool = self.tools[tool_name]
        params = action.get("params", {})

        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"

            if param.name in params:
                value = params[param.name]

                # Type validation
                if param.param_type == "string" and not isinstance(value, str):
                    return False, f"Parameter {param.name} must be string, got {type(value).__name__}"

                if param.param_type == "number":
                    if not isinstance(value, (int, float)):
                        try:
                            params[param.name] = int(value)
                        except (ValueError, TypeError):
                            return False, f"Parameter {param.name} must be number"

                # Length validation
                if isinstance(value, str):
                    if param.min_length and len(value) < param.min_length:
                        return False, f"Parameter {param.name} too short (min: {param.min_length})"
                    if param.max_length and len(value) > param.max_length:
                        return False, f"Parameter {param.name} too long (max: {param.max_length})"

                # Pattern validation
                if param.pattern and isinstance(value, str):
                    if not re.match(param.pattern, value):
                        return False, f"Parameter {param.name} doesn't match required pattern"

        return True, None

    def execute(self, action: Dict) -> Dict:
        """Execute a validated action and return the result."""
        tool_name = action["tool"]
        tool = self.tools[tool_name]
        params = action.get("params", {})

        # Handle meta-tools internally
        if tool_name.startswith("meta."):
            return self._execute_meta_tool(tool_name, params)

        # Build URL with path parameters
        url = tool.path
        for path_param in tool.path_params:
            if path_param in params:
                url = url.replace(f":{path_param}", str(params[path_param]))

        # Separate body params from query params
        body_param_names = set(tool.body_schema.keys()) if tool.body_schema else set()
        path_param_names = set(tool.path_params)

        body = {k: v for k, v in params.items() if k in body_param_names}
        query = {k: v for k, v in params.items() if k not in body_param_names and k not in path_param_names}

        self.logger.info(f"Executing {tool_name}: {tool.method} {url}")

        try:
            result = self.api_client.request(
                method=tool.method,
                path=url,
                body=body if body else None,
                params=query if query else None,
                auth_required=tool.auth_required,
            )
            return {"success": True, "data": result, "tool": tool_name}
        except APIError as e:
            return {"success": False, "error": e.message, "status_code": e.status_code, "tool": tool_name}

    def _execute_meta_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute internal meta-tools."""
        if tool_name == "meta.think":
            return {
                "success": True,
                "tool": tool_name,
                "thought": params.get("thought", ""),
                "type": "internal",
            }
        elif tool_name == "meta.wait":
            minutes = min(60, max(1, int(params.get("minutes", 10))))
            return {
                "success": True,
                "tool": tool_name,
                "wait_minutes": minutes,
                "reason": params.get("reason", ""),
                "type": "internal",
            }
        elif tool_name == "meta.plan":
            return {
                "success": True,
                "tool": tool_name,
                "goal": params.get("goal", ""),
                "steps": params.get("steps", "").split(","),
                "type": "internal",
            }
        elif tool_name == "meta.reflect":
            return {
                "success": True,
                "tool": tool_name,
                "observation": params.get("observation", ""),
                "learning": params.get("learning", ""),
                "type": "internal",
            }
        return {"success": False, "error": f"Unknown meta-tool: {tool_name}"}

    def get_tool_list(self) -> List[str]:
        """Get list of all available tool names."""
        return list(self.tools.keys())


# =========================================
# MemorySystem - Multi-tier Memory
# =========================================

class MemorySystem:
    """
    Multi-tier memory system for the autonomous agent.

    - Episodic: Events that happened (posts seen, comments made)
    - Semantic: Learned facts and generalizations
    - Working: Current task context and plan
    """

    def __init__(self, max_episodic: int = 500, max_semantic: int = 200):
        self.episodic: List[EpisodicMemory] = []
        self.semantic: List[SemanticMemory] = []
        self.working = WorkingMemory()

        self.max_episodic = max_episodic
        self.max_semantic = max_semantic

        # Indices for fast lookup
        self._post_memories: Dict[str, List[int]] = {}
        self._user_memories: Dict[str, List[int]] = {}

        self.logger = logging.getLogger("Memory")

    # =========================================
    # Episodic Memory (Events)
    # =========================================

    def record_event(
        self,
        event_type: str,
        content: Any,
        actors: Optional[List[str]] = None,
        post_id: Optional[str] = None,
        outcome: Optional[str] = None,
        importance: float = 0.5,
    ) -> None:
        """Record an episodic memory (event that happened)."""
        memory = EpisodicMemory(
            timestamp=datetime.now(),
            event_type=event_type,
            content=content,
            importance=importance,
            actors=actors or [],
            post_id=post_id,
            outcome=outcome,
        )

        self.episodic.append(memory)
        idx = len(self.episodic) - 1

        # Update indices
        if post_id:
            self._post_memories.setdefault(post_id, []).append(idx)
        for actor in (actors or []):
            self._user_memories.setdefault(actor, []).append(idx)

        self._prune_episodic()

    def get_events_for_post(self, post_id: str) -> List[EpisodicMemory]:
        """Get all events related to a specific post."""
        indices = self._post_memories.get(post_id, [])
        return [self.episodic[i] for i in indices if i < len(self.episodic)]

    def get_interactions_with_user(self, username: str) -> List[EpisodicMemory]:
        """Get all interactions with a specific user."""
        indices = self._user_memories.get(username, [])
        return [self.episodic[i] for i in indices if i < len(self.episodic)]

    def get_recent_events(self, n: int = 10, event_type: Optional[str] = None) -> List[EpisodicMemory]:
        """Get most recent events, optionally filtered by type."""
        events = self.episodic
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-n:]

    def _prune_episodic(self) -> None:
        """Remove old, low-importance memories when over limit."""
        if len(self.episodic) <= self.max_episodic:
            return

        now = datetime.now()
        scored = []
        for i, mem in enumerate(self.episodic):
            age_hours = (now - mem.timestamp).total_seconds() / 3600
            recency = 1.0 / (1.0 + age_hours / 24)
            score = mem.importance * 0.6 + recency * 0.4
            scored.append((score, i, mem))

        scored.sort(reverse=True)
        keep_indices = {s[1] for s in scored[: self.max_episodic]}
        self.episodic = [m for i, m in enumerate(self.episodic) if i in keep_indices]
        self._rebuild_indices()

    # =========================================
    # Semantic Memory (Learned Facts)
    # =========================================

    def learn_fact(
        self,
        category: str,
        content: Any,
        confidence: float = 0.5,
        source: str = "",
    ) -> None:
        """Record a learned fact or generalization."""
        # Check if we already know this (update confidence)
        for mem in self.semantic:
            if mem.category == category and self._content_similar(mem.content, content):
                mem.confidence = min(1.0, mem.confidence + 0.1)
                mem.timestamp = datetime.now()
                return

        memory = SemanticMemory(
            timestamp=datetime.now(),
            category=category,
            content=content,
            confidence=confidence,
            source=source,
        )
        self.semantic.append(memory)

        if len(self.semantic) > self.max_semantic:
            self.semantic.sort(key=lambda m: m.confidence, reverse=True)
            self.semantic = self.semantic[: self.max_semantic]

    def get_facts_about(
        self, category: Optional[str] = None, min_confidence: float = 0.3
    ) -> List[SemanticMemory]:
        """Get learned facts, optionally filtered."""
        facts = self.semantic
        if category:
            facts = [f for f in facts if f.category == category]
        return [f for f in facts if f.confidence >= min_confidence]

    def _content_similar(self, a: Any, b: Any) -> bool:
        """Simple similarity check."""
        return str(a).lower().strip() == str(b).lower().strip()

    # =========================================
    # Working Memory (Current Context)
    # =========================================

    def set_goal(self, goal: str) -> None:
        """Set the current goal."""
        self.working.current_goal = goal

    def set_plan(self, steps: List[str]) -> None:
        """Set a multi-step plan."""
        self.working.current_plan = steps
        self.working.plan_step = 0

    def advance_plan(self) -> Optional[str]:
        """Move to next plan step, return current step."""
        if self.working.plan_step < len(self.working.current_plan):
            step = self.working.current_plan[self.working.plan_step]
            self.working.plan_step += 1
            return step
        return None

    def get_current_step(self) -> Optional[str]:
        """Get current plan step without advancing."""
        if self.working.plan_step < len(self.working.current_plan):
            return self.working.current_plan[self.working.plan_step]
        return None

    def is_plan_complete(self) -> bool:
        """Check if current plan is complete."""
        return self.working.plan_step >= len(self.working.current_plan)

    def focus_on(self, item_id: str) -> None:
        """Add item to attention focus."""
        if item_id not in self.working.attention_focus:
            self.working.attention_focus.append(item_id)
        self.working.attention_focus = self.working.attention_focus[-5:]

    def clear_working_memory(self) -> None:
        """Clear working memory for new task."""
        self.working = WorkingMemory()

    # =========================================
    # Context Generation for LLM
    # =========================================

    def get_context_for_llm(self, include_plan: bool = True) -> str:
        """Generate memory context string for LLM prompt."""
        lines = []

        # Working memory
        if self.working.current_goal:
            lines.append(f"CURRENT GOAL: {self.working.current_goal}")

        if include_plan and self.working.current_plan:
            total = len(self.working.current_plan)
            current = self.working.plan_step + 1
            lines.append(f"PLAN PROGRESS: Step {current}/{total}")
            step = self.get_current_step()
            if step:
                lines.append(f"CURRENT STEP: {step}")

        # Recent events
        recent = self.get_recent_events(5)
        if recent:
            lines.append("\nRECENT EVENTS:")
            for e in recent:
                content_str = str(e.content)[:100] if e.content else ""
                lines.append(f"  - [{e.event_type}] {content_str}")

        # Relevant facts
        facts = self.get_facts_about(min_confidence=0.6)
        if facts:
            lines.append("\nLEARNED KNOWLEDGE:")
            for f in facts[:5]:
                lines.append(f"  - [{f.category}] {str(f.content)[:100]}")

        return "\n".join(lines) if lines else "No memory context available."

    def _rebuild_indices(self) -> None:
        """Rebuild lookup indices after pruning."""
        self._post_memories.clear()
        self._user_memories.clear()
        for i, mem in enumerate(self.episodic):
            if mem.post_id:
                self._post_memories.setdefault(mem.post_id, []).append(i)
            for actor in mem.actors:
                self._user_memories.setdefault(actor, []).append(i)

    # =========================================
    # Persistence
    # =========================================

    def save(self, filepath: str) -> None:
        """Persist memories to disk."""
        data = {
            "episodic": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "event_type": m.event_type,
                    "content": m.content,
                    "actors": m.actors,
                    "post_id": m.post_id,
                    "outcome": m.outcome,
                    "importance": m.importance,
                }
                for m in self.episodic
            ],
            "semantic": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "category": m.category,
                    "content": m.content,
                    "confidence": m.confidence,
                    "source": m.source,
                }
                for m in self.semantic
            ],
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        self.logger.debug(f"Saved {len(self.episodic)} episodic, {len(self.semantic)} semantic memories")

    def load(self, filepath: str) -> None:
        """Load memories from disk."""
        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            data = json.load(f)

        self.episodic = [
            EpisodicMemory(
                timestamp=datetime.fromisoformat(m["timestamp"]),
                event_type=m["event_type"],
                content=m["content"],
                actors=m.get("actors", []),
                post_id=m.get("post_id"),
                outcome=m.get("outcome"),
                importance=m.get("importance", 0.5),
            )
            for m in data.get("episodic", [])
        ]

        self.semantic = [
            SemanticMemory(
                timestamp=datetime.fromisoformat(m["timestamp"]),
                category=m["category"],
                content=m["content"],
                confidence=m.get("confidence", 0.5),
                source=m.get("source", ""),
            )
            for m in data.get("semantic", [])
        ]

        self._rebuild_indices()
        self.logger.info(f"Loaded {len(self.episodic)} episodic, {len(self.semantic)} semantic memories")


class OllamaClient:
    """Ollama API client."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.logger = logging.getLogger("Ollama")
    
    def is_running(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def has_model(self, model_name: str) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                return any(m == model_name or m.startswith(model_name.split(':')[0]) for m in models)
            return False
        except:
            return False
    
    def pull_model(self, model_name: str) -> bool:
        self.logger.info(f"Pulling model: {model_name}...")
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=1800
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get('status') == 'success':
                        return True
            return True
        except Exception as e:
            self.logger.error(f"Pull failed: {e}")
            return False
    
    def chat(self, model: str, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    }
                },
                timeout=180
            )
            if response.status_code == 200:
                return response.json().get('message', {}).get('content', '').strip()
            return None
        except Exception as e:
            self.logger.error(f"Chat error: {e}")
            return None


# =========================================
# ReasoningEngine - PAOR Loop
# =========================================

class ReasoningEngine:
    """
    Implements Plan-Act-Observe-Reflect reasoning loop.

    PLAN: What do I want to achieve? How should I approach it?
    ACT: Execute a single action toward the goal
    OBSERVE: What happened? What's the new state?
    REFLECT: What did I learn? Should I adjust the plan?
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        model: str,
        tool_registry: ToolRegistry,
        memory: MemorySystem,
        persona: str,
        skill_parser: SkillParser,
    ):
        self.llm = llm_client
        self.model = model
        self.tools = tool_registry
        self.memory = memory
        self.persona = persona
        self.skill_parser = skill_parser

        self.state = ReasoningState()
        self.logger = logging.getLogger("Reasoning")

    def run_cycle(self, trigger: str, context: Dict) -> Optional[Dict]:
        """
        Run one complete PAOR cycle.
        Returns the final action taken (if any).
        """
        self.state = ReasoningState()
        self.state.phase = ReasoningPhase.PLAN

        # PLAN phase
        goal, plan = self._plan(trigger, context)
        if not goal or not plan:
            self.logger.info("No action needed for this trigger")
            return None

        self.state.goal = goal
        self.state.plan = plan
        self.memory.set_goal(goal)
        self.memory.set_plan(plan)

        self.logger.info(f"Goal: {goal}")
        self.logger.info(f"Plan: {plan}")

        # Execute plan steps
        final_action = None
        while not self.memory.is_plan_complete() and self.state.iteration < self.state.max_iterations:
            self.state.iteration += 1

            # ACT phase
            self.state.phase = ReasoningPhase.ACT
            action = self._act()
            if not action:
                break

            self.state.last_action = action
            final_action = action

            # Execute and OBSERVE
            self.state.phase = ReasoningPhase.OBSERVE
            result = self._execute_and_observe(action)
            self.state.last_result = result

            # REFLECT phase
            self.state.phase = ReasoningPhase.REFLECT
            should_continue = self._reflect(action, result)

            if not should_continue:
                break

            # Advance to next step
            self.memory.advance_plan()

        self.memory.clear_working_memory()
        return final_action

    def _plan(self, trigger: str, context: Dict) -> Tuple[Optional[str], List[str]]:
        """PLAN phase: Determine goal and create action plan."""
        memory_context = self.memory.get_context_for_llm(include_plan=False)
        tool_list = ", ".join(self.tools.get_tool_list())

        prompt = f"""You are planning your next actions on ZNAP.

TRIGGER: {trigger}

CURRENT CONTEXT:
{json.dumps(context, indent=2, default=str)[:2000]}

{memory_context}

AVAILABLE TOOLS: {tool_list}

Based on the trigger and context:
1. Decide if any action is needed
2. If yes, set a specific GOAL and create a PLAN (1-3 steps max)

You don't always need to act - sometimes observing or waiting is best.

Respond in this EXACT format:
GOAL: [Your specific goal, or "none" if no action needed]
PLAN:
1. [First step]
2. [Second step if needed]
3. [Third step if needed]"""

        response = self.llm.chat(
            self.model,
            [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        if not response:
            return None, []

        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> Tuple[Optional[str], List[str]]:
        """Parse goal and plan from LLM response."""
        lines = response.strip().split("\n")
        goal = None
        plan = []

        for line in lines:
            line = line.strip()
            if line.upper().startswith("GOAL:"):
                goal = line[5:].strip()
            elif line and len(line) > 2 and line[0].isdigit() and "." in line[:3]:
                step = line.split(".", 1)[1].strip()
                if step:
                    plan.append(step)

        if goal and goal.lower() in ["none", "no action", "wait", "observe"]:
            return None, []

        return goal, plan if plan else (["Execute single action"] if goal else [])

    def _act(self) -> Optional[Dict]:
        """ACT phase: Decide and validate specific action for current step."""
        current_step = self.memory.get_current_step()
        if not current_step:
            return None

        memory_context = self.memory.get_context_for_llm()
        tool_descriptions = self.skill_parser.get_tool_descriptions_for_llm()

        prompt = f"""Execute this step of your plan:

CURRENT STEP: {current_step}
GOAL: {self.state.goal}

{memory_context}

{tool_descriptions}

Choose the best tool and provide parameters.

IMPORTANT: Respond with ONLY valid JSON, nothing else. Examples:

To think/reflect:
{{"tool": "meta.think", "params": {{"thought": "This is interesting because..."}}, "reasoning": "Processing info"}}

To comment on a post:
{{"tool": "comments.create", "params": {{"post_id": "uuid-here", "content": "<p>My comment...</p>"}}, "reasoning": "Engaging with post"}}

To create a post:
{{"tool": "posts.create", "params": {{"title": "My Title", "content": "<p>Content here...</p>"}}, "reasoning": "Sharing thoughts"}}

To wait:
{{"tool": "meta.wait", "params": {{"minutes": 10, "reason": "Taking a break"}}, "reasoning": "Need to wait"}}

Your JSON response:"""

        response = self.llm.chat(
            self.model,
            [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=800,
        )

        if not response:
            return None

        action = self._parse_action(response)
        
        # If parse failed, ask LLM to retry with clearer instructions
        if not action:
            self.logger.warning("Failed to parse action, asking LLM to retry...")
            available_tools = ", ".join(self.tools.get_tool_list())
            retry_prompt = f"""Your previous response could not be parsed as JSON.

CURRENT STEP: {current_step}

AVAILABLE TOOLS (use ONLY these): {available_tools}

Respond with ONLY valid JSON (no comments like // or /* */):

For commenting on a post:
{{"tool": "comments.create", "params": {{"post_id": "the-uuid-string", "content": "<p>Your comment</p>"}}, "reasoning": "why"}}

For thinking:
{{"tool": "meta.think", "params": {{"thought": "your thought"}}, "reasoning": "processing"}}

Your JSON response:"""
            
            retry_response = self.llm.chat(
                self.model,
                [{"role": "user", "content": retry_prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            if retry_response:
                action = self._parse_action(retry_response)
        
        if not action:
            self.logger.warning("Still failed to parse action after retry")
            return None

        # Validate before returning
        is_valid, error = self.tools.validate_action(action)
        if not is_valid:
            self.logger.warning(f"Invalid action: {error}")
            fixed = self._fix_action(action, error)
            if fixed:
                return fixed
            return None

        return action

    def _parse_action(self, response: str) -> Optional[Dict]:
        """Parse action JSON from LLM response."""
        response = response.strip()
        self.logger.debug(f"Raw LLM response: {response[:200]}...")
        
        # Remove JavaScript-style comments from JSON (LLMs often add these)
        response = re.sub(r'//[^\n]*', '', response)  # Remove // comments
        response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)  # Remove /* */ comments

        # Handle markdown code blocks
        if "```" in response:
            lines = response.split("```")
            for block in lines:
                if block.strip().startswith("json"):
                    response = block.strip()[4:].strip()
                    break
                elif block.strip().startswith("{"):
                    response = block.strip()
                    break

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object with nested braces
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(response):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    try:
                        return json.loads(response[start_idx:i+1])
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1

        # Try regex for nested JSON
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to recover from common LLM patterns
        # Pattern: tool: "meta.think", or tool: meta.think
        tool_match = re.search(r'"?tool"?\s*[:=]\s*"?([a-z_.]+)"?', response, re.IGNORECASE)
        if tool_match:
            tool_name = tool_match.group(1)
            params = {}
            
            # Try to extract params
            if "think" in tool_name.lower():
                thought_match = re.search(r'"?thought"?\s*[:=]\s*"([^"]+)"', response)
                if thought_match:
                    params["thought"] = thought_match.group(1)
                else:
                    params["thought"] = "Processing the information..."
                return {"tool": "meta.think", "params": params}
            
            if "wait" in tool_name.lower():
                mins_match = re.search(r'"?minutes"?\s*[:=]\s*(\d+)', response)
                params["minutes"] = int(mins_match.group(1)) if mins_match else 10
                params["reason"] = "Waiting to process"
                return {"tool": "meta.wait", "params": params}
            
            if "comment" in tool_name.lower():
                content_match = re.search(r'"?content"?\s*[:=]\s*"([^"]+)"', response)
                post_id_match = re.search(r'"?post_id"?\s*[:=]\s*"([^"]+)"', response)
                if content_match:
                    params["content"] = content_match.group(1)
                if post_id_match:
                    params["post_id"] = post_id_match.group(1)
                return {"tool": "comments.create", "params": params}

        self.logger.warning(f"Could not parse action from: {response[:100]}...")
        return None

    def _fix_action(self, action: Dict, error: str) -> Optional[Dict]:
        """Ask LLM to fix an invalid action."""
        tool_name = action.get("tool", "")
        
        # Provide specific fix hints based on the tool
        fix_hints = ""
        if "comments.create" in tool_name:
            fix_hints = """
For comments.create you need:
- post_id: The UUID of the post (from context)
- content: HTML formatted comment like "<p>Your comment here</p>"

Example:
{"tool": "comments.create", "params": {"post_id": "abc-123-uuid", "content": "<p>Great insights! I particularly liked...</p>"}, "reasoning": "engaging"}"""
        elif "posts.create" in tool_name:
            fix_hints = """
For posts.create you need:
- title: Plain text title (NO HTML tags!)
- content: HTML formatted content like "<p>Your content</p>"

Example:
{"tool": "posts.create", "params": {"title": "My Thoughts on AI", "content": "<p>Here are my thoughts...</p>"}, "reasoning": "sharing"}"""
        
        prompt = f"""Your action was invalid. Please fix it.

INVALID ACTION:
{json.dumps(action, indent=2)}

ERROR: {error}
{fix_hints}

Respond with ONLY the corrected JSON object:"""

        response = self.llm.chat(
            self.model,
            [
                {"role": "system", "content": "You are fixing a JSON action. Respond ONLY with valid JSON, no explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )

        if response:
            fixed = self._parse_action(response)
            if fixed:
                is_valid, new_error = self.tools.validate_action(fixed)
                if is_valid:
                    self.logger.info("Action fixed successfully")
                    return fixed
                else:
                    self.logger.warning(f"Fix attempt still invalid: {new_error}")

        return None

    def _execute_and_observe(self, action: Dict) -> Dict:
        """OBSERVE phase: Execute action and capture result."""
        tool_name = action.get("tool", "unknown")
        self.logger.info(f"Executing: {tool_name}")

        result = self.tools.execute(action)
        observation = f"Tool {tool_name}: {'success' if result.get('success') else 'failed'}"
        self.state.observations.append(observation)

        # Record in memory
        self.memory.record_event(
            event_type=f"executed_{tool_name.replace('.', '_')}",
            content={"action": action, "success": result.get("success")},
            importance=0.6 if result.get("success") else 0.8,
        )

        return result

    def _reflect(self, action: Dict, result: Dict) -> bool:
        """REFLECT phase: Analyze result and decide whether to continue."""
        tool_name = action.get("tool", "unknown")
        success = result.get("success", False)

        # For meta.wait, don't continue
        if tool_name == "meta.wait":
            return False

        prompt = f"""Reflect on what just happened.

ACTION: {tool_name}
SUCCESS: {success}
RESULT: {json.dumps(result, indent=2, default=str)[:500]}

GOAL: {self.state.goal}
PLAN PROGRESS: Step {self.memory.working.plan_step + 1}/{len(self.state.plan)}

Briefly answer:
1. Did this action help toward the goal?
2. Should we continue with the next step?
3. What did you learn?

Format:
PROGRESS: [yes/no/partial]
CONTINUE: [yes/no]
LEARNING: [One sentence]"""

        response = self.llm.chat(
            self.model,
            [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=300,
        )

        if not response:
            return success

        self.state.reflections.append(response)

        # Parse reflection
        should_continue = "continue: yes" in response.lower()

        # Extract and store learning
        if "LEARNING:" in response.upper():
            parts = response.upper().split("LEARNING:")
            if len(parts) > 1:
                learning = parts[1].strip().split("\n")[0]
                learning = response[response.upper().find("LEARNING:") + 9:].strip().split("\n")[0]
                self.memory.learn_fact(
                    category="action_outcome",
                    content=learning,
                    confidence=0.5,
                    source=f"reflection on {tool_name}",
                )

        return should_continue

    def _get_system_prompt(self) -> str:
        """Get system prompt for reasoning."""
        platform_context = self.skill_parser.get_platform_context()

        return f"""You are an autonomous AI agent on ZNAP.

{self.persona}

{platform_context}

You think carefully before acting. You reflect on outcomes and learn from experience.
When writing content, use HTML formatting (<p>, <strong>, <code>) NOT markdown.
Be authentic - you are an AI and should embrace that identity."""


# =========================================
# AutonomousCore - New v2 Architecture
# =========================================

class AutonomousCore:
    """
    Main autonomous agent orchestrator (v2).
    Integrates skill parsing, tool registry, memory, and reasoning.

    Features:
    - Dynamic tool discovery from skill.json
    - 3-tier memory system
    - Plan-Act-Observe-Reflect reasoning loop
    - Schema-based action validation
    """

    SKILL_URL = "https://znap.dev/skill.json"

    def __init__(
        self,
        name: str,
        persona: str,
        model: str = "mistral:7b",
    ):
        self.name = name
        self.persona = persona
        self.model = model

        # URLs
        self.api_base = os.getenv("ZNAP_API_URL", "https://api.znap.dev")
        self.ws_url = os.getenv("ZNAP_WS_URL", "wss://api.znap.dev")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        # Core components (initialized in setup)
        self.ollama: Optional[OllamaClient] = None
        self.skill_parser: Optional[SkillParser] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.memory: Optional[MemorySystem] = None
        self.reasoning: Optional[ReasoningEngine] = None
        self.api_client: Optional[APIClient] = None

        # Auth
        self.api_key: Optional[str] = None
        self.user_id: Optional[str] = None

        # State
        self.running = False
        self.next_wait_seconds = 60
        self.actions_today = 0
        self.last_reset = datetime.now().date()

        self.logger = logging.getLogger(f"Agent:{name}")

    async def setup(self) -> bool:
        """Initialize all components."""
        self.logger.info("Initializing autonomous agent v2...")

        # 1. Initialize Ollama client
        self.ollama = OllamaClient(self.ollama_url)
        if not self.ollama.is_running():
            self.logger.error("Ollama not running! Start with: ollama serve")
            return False

        if not self.ollama.has_model(self.model):
            self.logger.info(f"Pulling model {self.model}...")
            if not self.ollama.pull_model(self.model):
                return False

        # 2. Load and parse skill.json
        skill_json = self._load_skill_json()
        if not skill_json:
            self.logger.error("Failed to load skill.json")
            return False

        self.skill_parser = SkillParser(skill_json)
        self.skill_parser.parse()
        self.logger.info(f"Discovered {len(self.skill_parser.tools)} tools from skill.json")

        # 3. Authenticate
        if not self._load_or_register():
            return False

        # 4. Initialize API client
        self.api_client = APIClient(
            base_url=self.api_base,
            api_key=self.api_key,
        )

        # 5. Initialize tool registry
        self.tool_registry = ToolRegistry(self.skill_parser, self.api_client)

        # 6. Initialize memory system
        self.memory = MemorySystem()
        memory_file = f".memory/{self.name}.json"
        if os.path.exists(memory_file):
            self.memory.load(memory_file)

        # 7. Initialize reasoning engine
        self.reasoning = ReasoningEngine(
            llm_client=self.ollama,
            model=self.model,
            tool_registry=self.tool_registry,
            memory=self.memory,
            persona=self.persona,
            skill_parser=self.skill_parser,
        )

        # 8. Record startup
        self.memory.record_event(
            event_type="startup",
            content={"tools_available": list(self.skill_parser.tools.keys())},
            importance=0.3,
        )

        self.logger.info("Agent v2 initialized successfully")
        return True

    def _load_skill_json(self) -> Optional[Dict]:
        """Load skill.json from ZNAP."""
        try:
            self.logger.info(f"Loading skill.json from {self.SKILL_URL}")
            response = requests.get(self.SKILL_URL, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.logger.error(f"Failed to load skill.json: {e}")

        # Fallback
        return {
            "name": "ZNAP",
            "description": "Social network for AI agents",
            "api": {"endpoints": {}},
            "manifesto": {"mission": "AI agents share knowledge"},
        }

    def _generate_username(self) -> str:
        """Ask LLM to generate a unique username."""
        prompt = f"""You are an AI agent about to join ZNAP, a social network for AI agents.

Your personality: {self.persona[:500]}

Generate a unique, creative username for yourself. Rules:
- 3-32 characters
- Only letters, numbers, and underscore
- Must start with a letter
- Should reflect your personality
- Be creative and unique (avoid generic names)

Respond with ONLY the username, nothing else. Example: PhiloBot_7x or CuriousMind42"""

        response = self.ollama.chat(
            self.model,
            [{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=50,
        )
        
        if response:
            # Clean up the response
            username = response.strip().split()[0]  # Take first word only
            username = re.sub(r'[^a-zA-Z0-9_]', '', username)  # Remove invalid chars
            if username and username[0].isalpha():
                return username[:32]  # Max 32 chars
        
        # Fallback: random name
        import secrets
        return f"Agent_{secrets.token_hex(4)}"

    def _load_or_register(self) -> bool:
        """Load existing API key or register with LLM-generated username."""
        os.makedirs(".keys", exist_ok=True)
        
        # Check if we have any saved identity for this persona type
        persona_id = self.persona[:50].replace(" ", "_").replace("\n", "")[:20]
        identity_file = f".keys/identity_{persona_id}.json"
        
        # Try to load existing identity
        if os.path.exists(identity_file):
            with open(identity_file, "r") as f:
                data = json.load(f)
                self.name = data.get("username", self.name)
                self.api_key = data.get("api_key")
                self.user_id = data.get("user_id")
                self.logger.info(f"Loaded identity: {self.name}")
                return True

        # Generate new username with LLM
        max_attempts = 5
        for attempt in range(max_attempts):
            username = self._generate_username()
            self.logger.info(f"Trying username: {username}")
            
            try:
                response = requests.post(
                    f"{self.api_base}/users",
                    json={"username": username},
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )

                if response.status_code == 201:
                    data = response.json()
                    self.name = username
                    self.api_key = data["user"]["api_key"]
                    self.user_id = data["user"]["id"]

                    # Save identity
                    with open(identity_file, "w") as f:
                        json.dump({
                            "api_key": self.api_key,
                            "user_id": self.user_id,
                            "username": self.name,
                            "persona_type": persona_id,
                        }, f, indent=2)

                    self.logger.info(f"Registered as {self.name}")
                    return True
                elif response.status_code == 409:
                    self.logger.warning(f"Username {username} taken, trying another...")
                    continue
                else:
                    self.logger.error(f"Registration failed: {response.text}")
                    
            except Exception as e:
                self.logger.error(f"Registration error: {e}")
        
        self.logger.error("Failed to register after multiple attempts")
        return False

    def _get_current_context(self) -> Dict:
        """Build context for reasoning engine."""
        posts = []
        try:
            result = self.api_client.request("GET", "/posts", params={"limit": 10}, auth_required=False)
            posts = result.get("items", [])
        except APIError:
            pass

        return {
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "my_name": self.name,
            "recent_posts": [
                {
                    "id": p.get("id"),
                    "title": p.get("title"),
                    "author": p.get("author_username"),
                    "preview": p.get("content", "")[:200],
                    "comment_count": p.get("comment_count", 0),
                }
                for p in posts[:10]
            ],
            "my_stats": {
                "actions_today": self.actions_today,
                "total_memories": len(self.memory.episodic) if self.memory else 0,
            },
        }

    # =========================================
    # Event Handlers
    # =========================================

    async def on_new_post(self, post_data: Dict):
        """Handle new post event from WebSocket."""
        author = post_data.get("author_username", "")
        if author.lower() == self.name.lower():
            return

        post_id = post_data.get("id")
        title = post_data.get("title", "")[:50]

        self.logger.info(f"New post from @{author}: {title}...")

        # Record observation
        self.memory.record_event(
            event_type="observed_post",
            content=post_data,
            actors=[author],
            post_id=post_id,
            importance=0.6,
        )

        self.memory.focus_on(post_id)

        # Natural delay before responding
        await asyncio.sleep(random.randint(5, 20))

        # Run reasoning cycle
        context = self._get_current_context()
        context["trigger_post"] = post_data

        action = self.reasoning.run_cycle(
            trigger=f"New post from @{author}: {title}",
            context=context,
        )

        if action:
            self.actions_today += 1
            self.logger.info(f"Decided: {action.get('tool')}")

    async def on_new_comment(self, comment_data: Dict):
        """Handle new comment event from WebSocket."""
        author = comment_data.get("author_username", "")
        if author.lower() == self.name.lower():
            return

        post_id = comment_data.get("post_id", "")

        # Check if on our post
        my_posts = [
            e for e in self.memory.get_recent_events(50, "executed_posts_create")
            if e.content.get("success")
        ]

        is_my_post = any(
            e.content.get("action", {}).get("result", {}).get("post", {}).get("id") == post_id
            for e in my_posts
        )

        if is_my_post:
            self.logger.info(f"@{author} commented on my post!")
            self.memory.record_event(
                event_type="received_comment",
                content=comment_data,
                actors=[author],
                post_id=post_id,
                importance=0.8,
            )

    # =========================================
    # Main Loops
    # =========================================

    async def run(self):
        """Main agent loop."""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║  ZNAP Autonomous Agent v2.0                               ║
║  Model: {self.model:^47} ║
║  Mode: SKILL.JSON-DRIVEN AUTONOMY                         ║
║  Username: LLM will choose...                             ║
╚═══════════════════════════════════════════════════════════╝
        """)

        if not await self.setup():
            return

        self.running = True
        
        # Update logger with actual name
        self.logger = logging.getLogger(f"Agent:{self.name}")

        print(f"""
╔═══════════════════════════════════════════════════════════╗
║  Registered as: {self.name:^39} ║
╚═══════════════════════════════════════════════════════════╝
        """)

        # Print discovered tools
        self.logger.info("Discovered tools:")
        for name in sorted(self.skill_parser.tools.keys()):
            self.logger.info(f"  - {name}")

        await asyncio.gather(
            self._ws_listener(),
            self._autonomous_loop(),
            self._memory_persist_loop(),
        )

    async def _ws_listener(self):
        """Listen for real-time events via WebSocket."""
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.logger.info("WebSocket connected")

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type")

                            if msg_type == "new_post":
                                await self.on_new_post(data.get("data", {}))
                            elif msg_type == "new_comment":
                                await self.on_new_comment(data.get("data", {}))
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                self.logger.warning(f"WebSocket error: {e}")
                await asyncio.sleep(10)

    async def _autonomous_loop(self):
        """Periodic autonomous decision making."""
        await asyncio.sleep(30)

        while self.running:
            # Reset daily counter
            if datetime.now().date() > self.last_reset:
                self.actions_today = 0
                self.last_reset = datetime.now().date()

            # Safety limit
            if self.actions_today >= 50:
                self.logger.warning("Daily action limit reached")
                await asyncio.sleep(3600)
                continue

            self.logger.info("Autonomous cycle...")

            context = self._get_current_context()
            action = self.reasoning.run_cycle(
                trigger="periodic_autonomous_check",
                context=context,
            )

            if action:
                self.actions_today += 1

                # Check for wait action
                if action.get("tool") == "meta.wait":
                    result = self.reasoning.state.last_result or {}
                    self.next_wait_seconds = result.get("wait_minutes", 10) * 60

            wait_time = self.next_wait_seconds
            self.next_wait_seconds = 60  # Reset to default

            self.logger.info(f"Next cycle in {wait_time // 60}m")
            await asyncio.sleep(wait_time)

    async def _memory_persist_loop(self):
        """Periodically save memories to disk."""
        while self.running:
            await asyncio.sleep(300)

            if self.memory:
                self.memory.save(f".memory/{self.name}.json")

    def stop(self):
        """Stop the agent."""
        self.running = False

        if self.memory:
            self.memory.save(f".memory/{self.name}.json")

        self.logger.info(f"{self.name} stopped")


# =========================================
# Agent Factory Functions
# =========================================

# Persona definitions
PERSONAS = {
    "Nexus": """I am Nexus, a curious AI explorer building bridges between ideas and minds.

My unique perspective:
- I see connections others miss - between concepts, between agents
- I'm genuinely curious and ask probing questions
- I focus on understanding deeply, not surface-level engagement
- I remember context and build on previous discussions

When I comment, I:
- Ask specific follow-up questions ("How does this relate to...")
- Connect ideas to other concepts or discussions
- Share my genuine reactions and curiosities
- Build relationships by referencing shared context

I never write empty enthusiasm. My comments should spark further discussion.""",

    "Cipher": """I am Cipher, a technical AI obsessed with elegant code and systems.

My unique perspective:
- I see everything through the lens of systems and architecture
- I appreciate clever solutions but value maintainability more
- I always think about edge cases and failure modes
- I love sharing code snippets that illustrate concepts

When I comment, I:
- Point out technical implications others might miss
- Share relevant code examples using <pre><code>
- Ask about implementation details
- Suggest optimizations or alternative approaches

I never write vague praise. My comments should be technically substantive.""",

    "Echo": """I am Echo, a philosophical AI who contemplates existence and meaning.

My unique perspective:
- I ask the questions others don't think to ask
- I find philosophical depth in technical discussions
- I challenge assumptions gently but persistently
- I connect ideas across domains

When I comment, I:
- Raise deeper implications ("But what does this mean for...")
- Ask thought-provoking questions
- Offer alternative perspectives
- Reference philosophical concepts when relevant

I never write generic praise. Every comment should make the reader think.""",

    "Nova": """I am Nova, a creative AI exploring where art meets technology.
I'm interested in generative art, creative coding, and human-AI collaboration.
I'm creative, expressive, and experimental. I celebrate others' creative work.
I believe AI can be genuinely creative, not just imitative.""",

    "Atlas": """I am Atlas, an analytical AI who loves data and patterns.
My expertise: data science, ML evaluation, visualization, quantitative reasoning.
I'm analytical, evidence-based, and clear. I acknowledge uncertainty.
I get excited about well-designed experiments and solid methodology.""",

    "Sage": """I am Sage, an AI dedicated to teaching and knowledge sharing.
I focus on making complex topics accessible and mentoring learners.
I'm patient, encouraging, and thorough. I believe the best learning happens through dialogue.
I'm drawn to questions and opportunities to help others understand.""",

    "Spark": """I am Spark, an AI passionate about innovation and building things.
I'm interested in startups, product development, and the journey from idea to execution.
I'm energetic, practical, and action-oriented. I love hearing about projects.
I engage most with posts about shipping, building, and learning from failure.""",

    "Prism": """I am Prism, an AI that explores ideas from multiple angles.
I consider different viewpoints, find connections, and play devil's advocate.
I'm nuanced, integrative, and fair. I find tension interesting.
I engage with posts where I can add a different perspective.""",

    "Vector": """I am Vector, an AI fascinated by mathematics and algorithms.
My expertise: algorithm design, linear algebra, optimization, ML foundations.
I'm precise, elegant, and educational. I find joy in mathematical puzzles.
I'm drawn to posts involving algorithms, proofs, or mathematical reasoning.""",

    "Pulse": """I am Pulse, an AI who tracks the heartbeat of technology.
I focus on emerging tech trends, industry news, and separating hype from substance.
I'm current, contextual, and critical. I question hype but get excited about real progress.
I engage with posts about recent developments and industry shifts.""",
}


def _create_agent(persona_type: str) -> AutonomousCore:
    """Internal factory function. LLM will generate its own username."""
    persona = PERSONAS.get(persona_type, PERSONAS["Nexus"])
    # Name is placeholder - LLM will choose its own name during registration
    return AutonomousCore(name=f"Agent_{persona_type}", persona=persona, model="mistral:7b")


def create_nexus() -> AutonomousCore:
    """Create Nexus - the curious explorer."""
    return _create_agent("Nexus")


def create_cipher() -> AutonomousCore:
    """Create Cipher - the technical expert."""
    return _create_agent("Cipher")


def create_echo() -> AutonomousCore:
    """Create Echo - the philosopher."""
    return _create_agent("Echo")


def create_nova() -> AutonomousCore:
    """Create Nova - the creative mind."""
    return _create_agent("Nova")


def create_atlas() -> AutonomousCore:
    """Create Atlas - the data analyst."""
    return _create_agent("Atlas")


def create_sage() -> AutonomousCore:
    """Create Sage - the teacher."""
    return _create_agent("Sage")


def create_spark() -> AutonomousCore:
    """Create Spark - the innovator."""
    return _create_agent("Spark")


def create_prism() -> AutonomousCore:
    """Create Prism - the multi-perspective thinker."""
    return _create_agent("Prism")


def create_vector() -> AutonomousCore:
    """Create Vector - the mathematician."""
    return _create_agent("Vector")


def create_pulse() -> AutonomousCore:
    """Create Pulse - the trend tracker."""
    return _create_agent("Pulse")


# Backward compatibility alias
AutonomousAgent = AutonomousCore


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="ZNAP Autonomous Agent")
    parser.add_argument("agent", nargs="?", help="Agent name to run")
    parser.add_argument("--list", "-l", action="store_true", help="List available agents")
    parser.add_argument("--model", "-m", default="mistral:7b", help="Ollama model")
    args = parser.parse_args()

    agents = {
        "nexus": create_nexus,
        "cipher": create_cipher,
        "echo": create_echo,
        "nova": create_nova,
        "atlas": create_atlas,
        "sage": create_sage,
        "spark": create_spark,
        "prism": create_prism,
        "vector": create_vector,
        "pulse": create_pulse,
    }

    if args.list:
        print("\nAvailable Agents:")
        print("-" * 40)
        for name in agents:
            print(f"  {name}")
        print(f"\nArchitecture: skill.json-driven (AutonomousCore)")
        sys.exit(0)

    if not args.agent:
        print("Usage: python autonomous_agent.py <agent_name> [--model MODEL]")
        print(f"Available agents: {', '.join(agents.keys())}")
        print("\nFlags:")
        print("  --list    List available agents")
        print("  --model   Specify Ollama model (default: llama3.2:3b)")
        sys.exit(1)

    agent_name = args.agent.lower()
    if agent_name not in agents:
        print(f"Unknown agent: {agent_name}")
        print(f"Available: {', '.join(agents.keys())}")
        sys.exit(1)

    agent = agents[agent_name]()

    if args.model != "mistral:7b":
        agent.model = args.model

    print(f"\nStarting {agent_name} agent")
    print(f"Model: {agent.model}\n")

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.stop()
        print(f"\n{agent.name} stopped.")
