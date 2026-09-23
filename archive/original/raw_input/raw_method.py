from raw_all import *
if __name__=='__main__':
    with threadpool_limits(limits=1):execute(sys.argv[1])
