from __future__ import annotations

import argparse
import subprocess

import uvicorn

from app.mcp.server import run_stdio_server, run_streamable_http_server


def main() -> None:
    parser = argparse.ArgumentParser(prog="retrieval-api")
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve_parser = subcommands.add_parser("serve", help="Start the REST API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8000, type=int)

    subcommands.add_parser("mcp", help="Start stdio MCP server")

    mcp_http_parser = subcommands.add_parser(
        "mcp-http",
        help="Start HTTP/Streamable HTTP MCP server",
    )
    mcp_http_parser.add_argument("--host", default="127.0.0.1")
    mcp_http_parser.add_argument("--port", default=8301, type=int)
    mcp_http_parser.add_argument("--path", default="/mcp")

    health_parser = subcommands.add_parser("healthcheck", help="Check REST API health endpoint")
    health_parser.add_argument("--port", default=8000, type=int)

    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)
    elif args.command == "mcp":
        run_stdio_server()
    elif args.command == "mcp-http":
        run_streamable_http_server(host=args.host, port=args.port, path=args.path)
    elif args.command == "healthcheck":
        subprocess.run(
            [
                "python",
                "-c",
                "import urllib.request; "
                f"urllib.request.urlopen('http://127.0.0.1:{args.port}/v1/health', timeout=3)",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
