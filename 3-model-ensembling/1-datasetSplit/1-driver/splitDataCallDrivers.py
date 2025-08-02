import argparse

def driver_main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="split → call three drivers in threads → store possibilities in three files (+ store type for fit)"
    )
    # todo (use split and calculatePossibilities for transform)