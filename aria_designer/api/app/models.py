from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortModel(BaseModel):
    name: str
    dtype: str


class ComponentModel(BaseModel):
    id: str
    name: str
    category: str
    version: str = "v1"
    description: str = ""
    inputs: List[PortModel] = Field(default_factory=list)
    outputs: List[PortModel] = Field(default_factory=list)
    params_schema: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["draft", "approved", "deprecated"] = "draft"
    created_at: str = Field(default_factory=utc_now_iso)


class GraphNodeModel(BaseModel):
    id: str
    component_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    ui_meta: Dict[str, Any] = Field(default_factory=dict)


class GraphEdgeModel(BaseModel):
    id: str
    source: str
    source_port: str = "out"
    target: str
    target_port: str = "in"


class WorkflowGraphModel(BaseModel):
    schema_version: Literal["workflow_graph.v1"] = "workflow_graph.v1"
    workflow_id: str
    name: str
    nodes: List[GraphNodeModel]
    edges: List[GraphEdgeModel] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidateWorkflowRequest(BaseModel):
    workflow: WorkflowGraphModel
    selected_node_id: Optional[str] = None


class SuggestComponentsRequest(BaseModel):
    workflow: WorkflowGraphModel
    prompt: Optional[str] = None


class CompileWorkflowRequest(BaseModel):
    workflow: WorkflowGraphModel
    target: Literal["cpu", "cuda", "auto"] = "auto"


class RunWorkflowRequest(BaseModel):
    workflow: WorkflowGraphModel
    budget: Dict[str, Any] = Field(default_factory=dict)


class PatchOpModel(BaseModel):
    op: Literal["add_node", "remove_node", "replace_node", "rewire", "mutate_param"]
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class AriaPatchProposalModel(BaseModel):
    workflow_id: str
    base_version: int
    author: Literal["aria"] = "aria"
    rationale: str
    expected_impact: Dict[str, str] = Field(default_factory=dict)
    ops: List[PatchOpModel] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class ApplyPatchRequest(BaseModel):
    proposal_id: str
    approved_by: str

class RecordOutcomeRequest(BaseModel):
    suggestion_id: str
    outcome: Literal["applied", "rejected"]
    fingerprint: Optional[str] = None
    intent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class AskAriaPromptRequest(BaseModel):
    workflow: WorkflowGraphModel
    prompt: str
    base_version: int = 1


class ComponentConfigValidateRequest(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None


class ValidateWorkflowResponse(BaseModel):
    valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
