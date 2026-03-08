import ast
import os
import glob
from collections import defaultdict

def analyze_file(filepath):
    findings = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content, filename=filepath)
    except Exception as e:
        return findings

    has_math_import = False
    has_numba_import = False
    has_numpy_import = False
    has_torch_import = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'math': has_math_import = True
                if alias.name == 'numba': has_numba_import = True
                if alias.name == 'numpy': has_numpy_import = True
                if alias.name == 'torch': has_torch_import = True
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'math': has_math_import = True
            if node.module == 'numba': has_numba_import = True
            if node.module == 'numpy': has_numpy_import = True
            if node.module == 'torch': has_torch_import = True

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            has_slots = False
            is_exception = any("Exception" in getattr(base, 'id', '') for base in node.bases)
            is_torch_module = any("Module" in getattr(base, 'attr', getattr(base, 'id', '')) for base in node.bases)
            
            for child in node.body:
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == '__slots__':
                            has_slots = True
                            break
            
            if not has_slots and not is_exception and not is_torch_module:
                # We only suggest slots for simple classes (not PyTorch nn.Modules, not Exceptions)
                findings.append(f"Class '{node.name}' might benefit from __slots__")

    if has_math_import and not (has_numpy_import or has_torch_import):
        findings.append(f"Uses 'math' module but no 'numpy'/'torch' - potential vectorization opportunity")

    # Pure Python loops with arithmetic
    for node in ast.walk(tree):
        if isinstance(node, ast.For) or isinstance(node, ast.While):
            arithmetic_ops = 0
            for child in ast.walk(node):
                if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    arithmetic_ops += 1
            if arithmetic_ops > 5:
                findings.append(f"Heavy arithmetic loop detected at line {node.lineno} - candidate for Numba/Cython/Vectorization")

    return findings

if __name__ == '__main__':
    search_dirs = ['aria_core', 'aria_designer', 'HYDRA', 'LA3', 'research', 'AbstractMoE']
    all_findings = defaultdict(list)
    
    for sdir in search_dirs:
        for root, _, files in os.walk(sdir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    res = analyze_file(filepath)
                    if res:
                        all_findings[filepath].extend(res)
                        
    with open('research/PERF_FINDINGS.md', 'a') as f:
        for filepath, lines in all_findings.items():
            f.write(f"\n### {filepath}\n")
            for line in lines:
                f.write(f"- {line}\n")
