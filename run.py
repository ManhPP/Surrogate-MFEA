from mfea import mfea
from helpers import *
from tasks import *


def callback(res):
    pass


def main():
    config = load_config()

    for exp_id in range(config['repeat']):
        print('[+] MFEA - %d/%d' % (exp_id, config['repeat']))
        mfea(TS_TR(), config, callback)


if __name__ == '__main__':
    main()
