from fixture_transport import FixtureTransport

from bodhan_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main(transport=FixtureTransport()))
