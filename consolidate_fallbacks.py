
import os
import argparse
import torch

def consolidate_fallbacks(apply=False):
    base_dir = "aria_designer/components"
    updated_count = 0

    binary_ops_map = {
        'add': 'torch.add',
        'sub': 'torch.sub',
        'mul': 'torch.mul',
        'div_safe': 'lambda a, b: a / (b + 1e-6 * torch.where(b >= 0, 1.0, -1.0))',
    }

    unary_ops_map = {
        'abs': 'torch.abs',
        'exp': 'lambda x: torch.exp(torch.clamp(x, -20, 20))',
        'neg': 'torch.neg',
        'reciprocal': 'lambda x: 1.0 / torch.clamp(x, min=1e-8) if x.mean() > 0 else 1.0 / torch.clamp(x, max=-1e-8)',
        'square': 'torch.square',
    }

    for root, _, files in os.walk(base_dir):
        if 'kernel_fallback.py' in files:
            file_path = os.path.join(root, 'kernel_fallback.py')
            
            parts = root.split(os.sep)
            if 'components' not in parts: continue
            idx = parts.index('components')
            if len(parts) <= idx + 2: continue
            
            category = parts[idx + 1]
            component = parts[idx + 2]
            component_type = f"{category}/{component}"

            with open(file_path, 'r') as f:
                content = f.read()

            if 'make_unary_handler' in content or 'make_binary_handler' in content:
                continue

            new_content = None
            
            if component in binary_ops_map:
                op_call = binary_ops_map[component]
                # Check for standard binary pattern
                if ('class ComponentHandler' in content and 
                    ('inputs["a"]' in content or "inputs['a']" in content) and 
                    ('inputs["b"]' in content or "inputs['b']" in content)):
                    new_content = f'"""Fallback kernel shim for {component_type}."""\nimport torch\nfrom runtime.fallback_templates import make_binary_handler\n\nComponentHandler = make_binary_handler("{component_type}", {op_call})\n'
            
            elif component in unary_ops_map:
                op_call = unary_ops_map[component]
                # Check for standard unary pattern
                if ('class ComponentHandler' in content and 
                    ('inputs["x"]' in content or "inputs['x']" in content)):
                    new_content = f'"""Fallback kernel shim for {component_type}."""\nimport torch\nfrom runtime.fallback_templates import make_unary_handler\n\nComponentHandler = make_unary_handler("{component_type}", {op_call})\n'

            if new_content:
                print(f"Updating {file_path} (Type: {component_type})")
                updated_count += 1
                if apply:
                    with open(file_path, 'w') as f:
                        f.write(new_content)

    print(f"Total files updated: {updated_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    consolidate_fallbacks(apply=args.apply)
