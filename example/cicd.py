#


###########################################################################
# Steps
###########################################################################

def load():
    from example.load import main
    main()


def preprocess():
    from example.preprocess import main
    main()


def defaults():
    from example.defaults import main
    main()


def categoricals():
    from example.categoricals import main
    main()


def lookups():
    from example.lookups import main
    main()


def train():
    from example.train import main
    main()


def evaluate():
    from example.evaluate import main
    main()


###########################################################################
# Main
###########################################################################

def main():
    load()
    preprocess()
    categoricals()
    lookups()
    train()
    evaluate()


if __name__ == "__main__":
    main()
