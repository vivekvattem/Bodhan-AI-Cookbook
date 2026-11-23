from bodhan_eval.cli import main
from fixture_transport import FixtureTransport

if __name__ == "__main__":
    raise SystemExit(main(transport=FixtureTransport()))
