"""仅加载指定上游函数，避开训练/评测依赖；不改动第三方源码。"""
import ast
from pathlib import Path


def load_functions(path, names, namespace, *, cpu_rotation=False):
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise RuntimeError(f'上游函数结构改变，请检查 {path}: {names}')
    if cpu_rotation:
        # 唯一设备适配：SVD 已在 CPU 进行，省去最后的 R.cuda()。
        target = selected[0]
        removed = [n for n in target.body if ast.unparse(n) == 'R = R.cuda()']
        if len(removed) != 1:
            raise RuntimeError(f'上游旋转函数变化，需要重新核对: {path}')
        target.body = [n for n in target.body if n not in removed]
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return [namespace[name] for name in names]
