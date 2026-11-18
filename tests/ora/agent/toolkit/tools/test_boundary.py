"""First static boundary guard; this is deliberately not a process sandbox."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]/'ora/agent/toolkit/tools'
HOST = 'ora.agent.toolkit.host'
ALLOWED = {'ast', '__future__', 'types', 'typing', 'dataclasses', 'collections', 'functools', 'itertools',
           'math', 'time', 'json', 'hashlib', 'base64', 're', 'enum', 'numpy', 'scipy'}
FORBIDDEN = {'open', 'read_bytes', 'read_text', 'write_bytes', 'write_text', 'mkdir', 'unlink',
             'rename', 'replace', 'socket', 'connect', 'urlopen', 'request', 'getenv', 'system',
             'popen', 'eval', 'exec', '__import__', 'import_module', 'register', 'reference'}


def test_tools_only_use_public_interface_lib_and_tools():
    for path in ROOT.rglob('*.py'):
        package = 'ora.agent.toolkit.tools' + ('.' + '.'.join(path.relative_to(ROOT).parts[:-1]) if path.parent != ROOT else '')
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split('.')[0] in ALLOWED, (path, alias.name)
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split('.')
                    prefix = '.'.join(parts[:len(parts) - node.level + 1])
                    module = prefix + ('.' + node.module if node.module else '')
                    assert any(module == root or module.startswith(root + '.') for root in
                               ('ora.agent.toolkit.tools', 'ora.agent.toolkit.lib', 'ora.agent.toolkit.interface')), (path, module)
                else:
                    assert node.module.split('.')[0] in ALLOWED, (path, node.module)
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
                assert name not in FORBIDDEN, (path, name)
            if isinstance(node, ast.Attribute):
                assert node.attr not in {'store', 'adapters', 'tool_data', '__dict__', '__globals__', '__subclasses__'}, (path, node.attr)


def test_tools_lib_and_interface_never_import_the_host():
    """The trusted host (ora/agent/toolkit/host/) runs the tools; nothing under tools/, lib/ or interface/ imports it."""
    toolkit = ROOT.parent
    for top in ('tools', 'lib', 'interface'):
        for path in (toolkit/top).rglob('*.py'):
            package = 'ora.agent.toolkit.' + '.'.join(path.relative_to(toolkit).parts[:-1])
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if node.level:
                        parts = package.split('.')
                        module = '.'.join(parts[:len(parts) - node.level + 1] + ([module] if module else []))
                    names = [module] + [module + '.' + alias.name for alias in node.names]
                else:
                    continue
                assert not any(name == HOST or name.startswith(HOST + '.') for name in names), (path, names)
