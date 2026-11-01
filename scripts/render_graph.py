"""Write the compiled workflow as Mermaid to docs/graph.mmd. No model server needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragdocs.config import load_config, resolve
from ragdocs.graph import build_graph, render_mermaid
from ragdocs.retrieval import Index

if __name__ == "__main__":
    cfg = load_config()
    index = Index(resolve(cfg, cfg["index"]["path"]), None, False)
    graph = build_graph(cfg, index, llm=None)  # rendering does not call the model
    out = Path(__file__).resolve().parents[1] / "docs" / "graph.mmd"
    out.parent.mkdir(exist_ok=True)
    out.write_text(render_mermaid(graph), encoding="utf-8")
    print(f"wrote {out}")
