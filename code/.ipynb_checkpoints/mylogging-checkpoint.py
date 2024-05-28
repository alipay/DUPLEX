import logging
import os
import time
# import args

def getLogger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    sHandler.setLevel(logging.INFO)
    logger.addHandler(sHandler)
    if args.save_log:
    # FileHandler
        work_dir = os.path.join(args.log_path,
                                time.strftime("%Y-%m-%d-%H.%M", time.localtime())+'%s.txt'%(args.save_log)) 
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        fHandler = logging.FileHandler(work_dir, mode='w')
        fHandler.setLevel(logging.DEBUG) 
        fHandler.setFormatter(formatter) 
        logger.addHandler(fHandler) 

    return logger
