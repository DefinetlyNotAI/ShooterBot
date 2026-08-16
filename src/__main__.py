from src.noise_control import configure_library_noise

configure_library_noise()

from src.cli import run

# allow python -m src
if __name__ == "__main__":
    run.main()
