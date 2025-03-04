import logging

def get_logger(name="app", log_file="app.log", level=logging.INFO):
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 추가된 경우 중복 추가 방지
    if not logger.hasHandlers():
        logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
