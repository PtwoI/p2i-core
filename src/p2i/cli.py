"""Compatibility entry point; command implementation lives in p2i-cli."""


def main():
    try:
        from p2i_cli.main import main as cli_main
    except ImportError as exc:
        raise RuntimeError("Install p2i-cli to use terminal commands") from exc
    return cli_main()


if __name__ == "__main__":
    main()
