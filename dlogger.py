import torch
import logging
import torch.distributed as dist
from dutil import get_lrank


class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank  # 添加 rank 信息到日志记录的记录对象
        return True


class DLogger:
    def __init__(self, name="DLogger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        lrank=get_lrank()
        formatter = logging.Formatter('rank:%(rank)s - %(message)s')
        ch.setFormatter(formatter)

        # 将 RankFilter 添加到处理器
        rank_filter = RankFilter(str(lrank))
        ch.addFilter(rank_filter)

        self.logger.addHandler(ch)
        self.is_master = False
        if lrank == 0:
            self.is_master = True
    def pstr(self, *args):
        msg = ' '.join(str(arg) for arg in args)
        return msg
        

    def info(self, *args, master_only=False):
        msg = self.pstr(*args)
        if master_only and not self.is_master:
            return
        self.logger.info(msg)

    def warning(self, *args, master_only=False):
        msg = self.pstr(*args)
        if master_only and not self.is_master:
            return
        self.logger.warning(msg)

    def error(self,*args, master_only=False):
        msg = self.pstr(*args)
        if master_only and not self.is_master:
            return
        self.logger.error(msg)