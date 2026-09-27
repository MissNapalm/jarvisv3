"""Free, fully local brain: the same tools and prompt as the Claude brain, driven by a model in Ollama."""
from __future__ import annotations

import json

import httpx

from .brain import SYSTEM_TEMPLATE, build_tools
from .library import Library, pretty
from .player import Player

DEFAULT_MODEL = "qwen3:14b"
MAX_HISTORY_MESSAGES = 40
MAX_TOOL_ROUNDS = 8
# Smaller local models need the tool contract spelled out more firmly than Claude does.
LOCAL_RULES = """

Rules for using tools:
- Every request to play, continue, pause, skip, seek, stop or change volume MUST be done by calling a tool.
  Never say something is playing, paused or stopped unless a tool result in this turn confirms it.
- To continue or resume a show, call continue_series with its exact series key; do not stop at find_series.
- For questions about how many shows or episodes there are, count from the tool results, not from memory."""


class LocalBrain:
    def __init__(self, lib: Library, player: Player, model: str = DEFAULT_MODEL,
                 host: str = "http://127.0.0.1:11434", think: bool = True):
        self.model = model
        self.think = think  # reasoning before acting: slower, but far fewer "said it but didn't do it" slips
        self.http = httpx.Client(base_url=host, timeout=120)
        self.tools = {t.name: t for t in build_tools(lib, player)}
        self.tool_specs = [{"type": "function", "function": {
            "name": t.name, "description": t.to_dict()["description"],
            "parameters": t.to_dict()["input_schema"]}} for t in self.tools.values()]
        series = lib.all_series()
        self.system = SYSTEM_TEMPLATE.format(
            n_series=len(series), series_list="\n".join(f"- {pretty(s)} [{s}]" for s in series)) + LOCAL_RULES
        self.messages: list[dict] = []
        self.http.post("/api/chat", json={"model": model, "messages": [], "keep_alive": "30m"})  # load into VRAM now

    def reset(self) -> None:
        self.messages = []

    def note(self, user_text: str, reply: str) -> None:
        """Record a command handled outside the model so later requests have the context."""
        if self.messages and self.messages[-1]["role"] == "user":
            return  # keep roles alternating; drop the note rather than break the history
        self.messages += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]

    def _trim(self) -> None:
        while len(self.messages) > MAX_HISTORY_MESSAGES:
            cut = next((i for i, m in enumerate(self.messages[1:], 1) if m["role"] == "user"), None)
            if cut is None:
                break
            self.messages = self.messages[cut:]

    def ask(self, text: str, on_tool=None) -> str:
        self.messages.append({"role": "user", "content": text})
        reply = ""
        for _ in range(MAX_TOOL_ROUNDS):
            r = self.http.post("/api/chat", json={
                "model": self.model, "stream": False, "think": self.think, "keep_alive": "30m",
                "messages": [{"role": "system", "content": self.system}, *self.messages],
                "tools": self.tool_specs,
                "options": {"num_ctx": 16384, "temperature": 0.2},
            })
            r.raise_for_status()
            msg = r.json()["message"]
            self.messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            calls = msg.get("tool_calls") or []
            reply = (msg.get("content") or "").strip()
            if not calls:
                break
            for call in calls:
                name, args = call["function"]["name"], call["function"].get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                if on_tool:
                    on_tool(name, args)
                tool = self.tools.get(name)
                try:
                    result = tool.call(args) if tool else f"Error: no tool named {name}"
                except Exception as e:  # hand errors back to the model so it can recover
                    result = f"Error: {e}"
                self.messages.append({"role": "tool", "tool_name": name, "content": str(result)})
        self._trim()
        return reply or "Done."
