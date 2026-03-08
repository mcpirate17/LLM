from __future__ import annotations

import logging
import importlib.util
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from .. import database as db
from ..loader import COMPONENTS_ROOT
from ..models import ComponentModel, ComponentConfigValidateRequest, utc_now_iso as _utc_now

# Try to import runtime bridge from both import styles.
try:
    from runtime.bridge import get_component_execution_capability as bridge_component_capability, list_available_primitives as bridge_list_primitives
    HAS_BRIDGE = True
except ImportError:
    try:
        from aria_designer.runtime.bridge import get_component_execution_capability as bridge_component_capability, list_available_primitives as bridge_list_primitives
        HAS_BRIDGE = True
    except ImportError:
        bridge_component_capability = None
        bridge_list_primitives = None
        HAS_BRIDGE = False

from ..marketplace import search_marketplace, install_component
from ..benchmark_targets import benchmark_target_catalog

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["components"])

# ── Lookup helpers ────────────────────────────────────────────────────

def _require_component(component_id: str):
    comp = db.get_component(component_id)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"Component {component_id} not found")
    return comp

# ── Health ────────────────────────────────────────────────────────────

@router.get("/health")
def health() -> Dict[str, Any]:
    counts = db.count_components()
    return {"status": "ok", "components": counts}

# ── Components ────────────────────────────────────────────────────────

@router.get("/components")
def list_components(
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="Filter by status (default: approved)"),
) -> List[Dict[str, Any]]:
    """List registered components. Defaults to approved only."""
    if status is None:
        status = "approved"
    return db.list_components(category=category, status=status)

@router.get("/primitives")
def list_primitives() -> List[Dict[str, Any]]:
    """List all available primitives from the research pipeline."""
    if not HAS_BRIDGE or bridge_list_primitives is None:
        raise HTTPException(status_code=501, detail="Research bridge not available")
    return bridge_list_primitives()

@router.get("/marketplace/search")
def get_marketplace_search(q: str = Query(...)) -> List[Dict[str, Any]]:
    return search_marketplace(q)

@router.post("/marketplace/install")
def post_marketplace_install(req: Dict[str, str]) -> Dict[str, Any]:
    component_id = req.get("component_id")
    if not component_id:
        raise HTTPException(status_code=400, detail="Missing component_id")
    return install_component(component_id)

@router.get("/components/{component_id}")
def get_component(component_id: str) -> Dict[str, Any]:
    """Get a single component by ID."""
    comp = _require_component(component_id)
    return comp

@router.get("/components/{component_id}/properties")
def get_component_properties(component_id: str) -> Dict[str, Any]:
    """Return normalized property schema/defaults for one component."""
    comp = _require_component(component_id)

    params = comp.get("params") or {}
    properties = []
    for name, schema in params.items():
        schema = schema or {}
        properties.append({
            "name": name,
            "type": schema.get("type", "string"),
            "default": schema.get("default"),
            "description": schema.get("description", ""),
            "options": schema.get("options"),
            "constraints": schema.get("constraints"),
            "format": schema.get("format"),
            "required": bool(schema.get("required", False)),
        })

    return {
        "component_id": comp.get("id"),
        "component_name": comp.get("name"),
        "category": comp.get("category"),
        "description": comp.get("description", ""),
        "inputs": comp.get("inputs", []),
        "outputs": comp.get("outputs", []),
        "properties": properties,
    }

@router.get("/components/{component_id}/execution-capability")
def get_component_execution_capability(component_id: str) -> Dict[str, Any]:
    """Return execution capability across native/runtime bridge paths."""
    comp = _require_component(component_id)

    category = comp.get("category", "")
    manifest_id = comp.get("id", component_id)
    component_type = f"{category}/{manifest_id}" if category else manifest_id

    component_dir = COMPONENTS_ROOT / str(category) / str(manifest_id)
    native_impl = []
    if (component_dir / "kernel.c").exists():
        native_impl.append("c")
    if (component_dir / "kernel.cpp").exists() or (component_dir / "kernel.cc").exists():
        native_impl.append("cpp")
    if (component_dir / "kernel.rs").exists():
        native_impl.append("rust")
    if (component_dir / "kernel.pyx").exists():
        native_impl.append("cython")

    python_fallback = (component_dir / "kernel_fallback.py").exists()

    bridge_info: Dict[str, Any] = {
        "bridge_supported": False,
        "primitive_name": None,
        "execution_class": "unknown",
        "reason": "Research bridge unavailable in this environment.",
    }
    if HAS_BRIDGE and bridge_component_capability:
        try:
            bridge_info = bridge_component_capability(component_type)
        except Exception as exc:
            bridge_info = {
                "bridge_supported": False,
                "primitive_name": None,
                "execution_class": "unknown",
                "reason": f"Capability check failed: {exc}",
            }

    return {
        "component_id": manifest_id,
        "component_type": component_type,
        "category": category,
        "native_impl": native_impl,
        "python_fallback": python_fallback,
        "preferred_backend": native_impl[0] if native_impl else ("python" if python_fallback else "none"),
        "bridge": bridge_info,
        "has_semantic_warnings": bool(bridge_info.get("warnings")),
    }

@router.get("/integration/bridge-gap-report")
def get_bridge_gap_report() -> Dict[str, Any]:
    """Summarize components unsupported by the research primitive bridge."""
    comps = db.list_components(status="approved")
    gaps: List[Dict[str, Any]] = []
    by_class: Dict[str, int] = {}
    by_category: Dict[str, int] = {}

    for comp in comps:
        cid = comp.get("id")
        category = comp.get("category", "")
        ctype = f"{category}/{cid}" if category else str(cid)
        cap = (
            bridge_component_capability(ctype)
            if HAS_BRIDGE and bridge_component_capability
            else {
                "bridge_supported": False,
                "execution_class": "unknown",
                "reason": "Research bridge unavailable in this environment.",
                "primitive_name": None,
            }
        )
        if cap.get("bridge_supported"):
            continue

        execution_class = str(cap.get("execution_class", "unknown"))
        by_class[execution_class] = by_class.get(execution_class, 0) + 1
        by_category[category] = by_category.get(category, 0) + 1
        gaps.append(
            {
                "component_id": cid,
                "component_type": ctype,
                "category": category,
                "execution_class": execution_class,
                "reason": cap.get("reason", ""),
            }
        )

    gaps.sort(key=lambda row: (row["category"], row["component_id"]))
    return {
        "total_components": len(comps),
        "unsupported_components": len(gaps),
        "by_execution_class": dict(sorted(by_class.items())),
        "by_category": dict(sorted(by_category.items())),
        "gaps": gaps,
    }


@router.get("/benchmarks/targets")
def get_benchmark_targets_alias() -> Dict[str, Any]:
    """Alias route for benchmark target catalog at /api/v1/benchmarks/targets."""
    return benchmark_target_catalog()

@router.post("/components/{component_id}/validate-config")
def validate_component_config(component_id: str, req: ComponentConfigValidateRequest) -> Dict[str, Any]:
    """Validate a component config payload against manifest param schema/defaults."""
    comp = _require_component(component_id)

    params = comp.get("params") or {}
    raw_config = req.config or {}
    normalized = {}
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    def _type_ok(schema: Dict[str, Any], value: Any) -> bool:
        expected = schema.get("type")
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "float":
            return (isinstance(value, (int, float)) and not isinstance(value, bool))
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "string":
            return isinstance(value, str)
        if expected == "enum":
            if schema.get("multi_select") or schema.get("multiple"):
                if not isinstance(value, (list, tuple)):
                    return False
                return all(isinstance(v, (str, int, float, bool)) for v in value)
            return isinstance(value, (str, int, float, bool))
        return True

    for name, schema in params.items():
        schema = schema or {}
        has_value = name in raw_config
        value = raw_config.get(name, schema.get("default"))
        normalized[name] = value

        if schema.get("required", False) and (value is None or value == ""):
            errors.append({"param": name, "message": "Required parameter is missing"})
            continue

        if value is None:
            continue

        expected_type = schema.get("type")
        if expected_type and not _type_ok(schema, value):
            errors.append({
                "param": name,
                "message": f"Expected {expected_type}, got {type(value).__name__}",
            })
            continue

        if expected_type == "enum":
            options = schema.get("options") or []
            if options:
                if schema.get("multi_select") or schema.get("multiple"):
                    invalid_values = [v for v in (value or []) if v not in options]
                    if invalid_values:
                        errors.append({
                            "param": name,
                            "message": f"Invalid options {invalid_values}. Allowed: {options}",
                        })
                        continue
                elif value not in options:
                    errors.append({
                        "param": name,
                        "message": f"Invalid option '{value}'. Allowed: {options}",
                    })
                    continue

        constraints = schema.get("constraints") or {}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            min_v = constraints.get("min")
            max_v = constraints.get("max")
            if min_v is not None and value < min_v:
                errors.append({"param": name, "message": f"Must be >= {min_v}"})
            if max_v is not None and value > max_v:
                errors.append({"param": name, "message": f"Must be <= {max_v}"})

        if not has_value and schema.get("default") is not None:
            warnings.append({"param": name, "message": "Using default value"})

    for name in raw_config.keys():
        if name not in params:
            warnings.append({"param": name, "message": "Unknown parameter for this component"})

    category = str(comp.get("category") or "")
    manifest_id = str(comp.get("id") or component_id)
    fallback_path = COMPONENTS_ROOT / category / manifest_id / "kernel_fallback.py"
    if fallback_path.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                f"validate_handler_{category}_{manifest_id}",
                str(fallback_path),
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                handler_cls = getattr(module, "ComponentHandler", None)
                if handler_cls is not None:
                    handler = handler_cls()
                    validate_fn = getattr(handler, "validate_config", None)
                    if callable(validate_fn):
                        custom_errors = validate_fn(normalized) or []
                        for msg in custom_errors:
                            errors.append({"param": "__component__", "message": str(msg)})
        except Exception as exc:
            warnings.append({
                "param": "__component__",
                "message": f"Custom validation unavailable: {exc}",
            })

    return {
        "component_id": comp.get("id"),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "normalized_config": normalized,
    }

@router.get("/components/property-audit/report")
def get_component_property_audit() -> Dict[str, Any]:
    """Audit property coverage/defaults/help for all components."""
    from ..property_audit import audit_components
    return audit_components(COMPONENTS_ROOT)

@router.post("/components")
def create_component(component: ComponentModel) -> Dict[str, Any]:
    """Register a new component (status=draft)."""
    manifest = component.model_dump()
    if "params" not in manifest:
        manifest["params"] = manifest.get("params_schema") or {}
    manifest["status"] = "draft"
    now = _utc_now()
    db.upsert_component(manifest, created_at=now, updated_at=now)
    return manifest

@router.post("/components/{component_id}/approve")
def approve_component(component_id: str) -> Dict[str, str]:
    """Approve a component for use in the palette."""
    if not db.update_component_status(component_id, "approved", _utc_now()):
        raise HTTPException(status_code=404, detail="Component not found")
    return {"status": "approved", "component_id": component_id}

@router.post("/components/{component_id}/deprecate")
def deprecate_component(component_id: str) -> Dict[str, str]:
    """Deprecate a component (hidden from new workflows)."""
    if not db.update_component_status(component_id, "deprecated", _utc_now()):
        raise HTTPException(status_code=404, detail="Component not found")
    return {"status": "deprecated", "component_id": component_id}

@router.post("/components/reload")
def reload_components() -> Dict[str, Any]:
    """Re-scan components/ directory and reload into DB."""
    from ..loader import scan_and_load
    count = scan_and_load()
    return {"status": "ok", "reloaded": count, "count": count}
