"""Provider tape replay and full request comparison for tool-dispatch parity diagnostics."""
from pathlib import Path


def register(commands):
    parser = commands.add_parser("provider-replay", help="serve recorded model replies or compare complete requests")
    sub = parser.add_subparsers(dest="provider_replay_command", required=True)
    server = sub.add_parser("serve", help="serve a read-only reply tape; does not start ORA")
    server.add_argument("source", type=Path)
    server.add_argument("--output", type=Path, required=True)
    server.add_argument("--listen", default="127.0.0.1")
    server.add_argument("--port", type=int, default=18080)
    server.add_argument("--ready-file", type=Path, help="atomically publish the bound endpoint (port 0 supported)")
    compare = sub.add_parser("compare", help="compare every request, with per-image PSNR")
    compare.add_argument("source", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, required=True)
