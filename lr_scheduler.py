import math
from torch.optim.lr_scheduler import _LRScheduler


# class Anneals_Cosine_LR_Scheduler(_LRScheduler):
#     def __init__(self, optimizer, init_lr_=0.0, max_lr_=1.5e-4, min_lr_=1e-5, lr_warmup_steps_=3200, now_steps_=0, lr_decay_steps_=320000):     
#         self.init_lr_ = init_lr_
#         self.max_lr_ = max_lr_
#         self.min_lr_ = min_lr_
#         self.lr_warmup_steps_ = lr_warmup_steps_
#         self.now_steps_ = now_steps_
#         self.lr_decay_steps_ = lr_decay_steps_
#         super(Anneals_Cosine_LR_Scheduler, self).__init__(optimizer=optimizer)

#     def get_lr(self):
#         # Use linear warmup for the initial part
#         if self.lr_warmup_steps_ > 0 and self.now_steps_ <= self.lr_warmup_steps_:
#             self.now_steps_ += 1
#             return [(self.init_lr_ + (self.max_lr_ - self.init_lr_) * self.now_steps_ / self.lr_warmup_steps_) for _ in self.optimizer.param_groups]
#         # For any steps larger than lr_decay_steps, use self.min_lr
#         if self.now_steps_ > self.lr_decay_steps_:
#             self.now_steps_+= 1
#             return [self.min_lr_ for _ in self.optimizer.param_groups]
#         # If we are done with the warmup period, use the consine decay style
#         num_step = self.now_steps_ - self.lr_warmup_steps_ 
#         decay_step = self.lr_decay_steps_ - self.lr_warmup_steps_ 
#         decay_ratio = num_step / decay_step 
#         delta_lr = self.max_lr_ - self.min_lr_ 
#         coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0) 
#         self.now_steps_ += 1 
#         return [self.min_lr_ + coeff * delta_lr  for _ in self.optimizer.param_groups]


class Anneals_Cosine_LR_Scheduler:
    def __init__(self,init_lr=0.0,max_lr=1.5e-4,min_lr=1e-5,lr_warmup_steps=700,lr_decay_steps=320000):
        self.init_lr_ =init_lr
        self.max_lr_ =max_lr
        self.min_lr_ = min_lr

        self.lr_warmup_steps_ = lr_warmup_steps
        self.now_steps_ =0
        self.lr_decay_steps_ =lr_decay_steps

    def step_lr(self):
        if self.lr_warmup_steps_>0 and self.now_steps_ <= self.lr_warmup_steps_:
            self.now_steps_+=1
            return self.init_lr_+(self.max_lr_-self.init_lr_)*self.now_steps_/self.lr_warmup_steps_
        
        if self.now_steps_> self.lr_decay_steps_:
            self.now_steps_+=1
            return self.min_lr_
        
        num_step=self.now_steps_-self.lr_warmup_steps_
        decay_step = self.lr_decay_steps_-self.lr_warmup_steps_
        decay_ratio = num_step/decay_step
        delta_lr = self.max_lr_ - self.min_lr_
        coeff = 0.5*(math.cos(math.pi*decay_ratio)+1.0)
        self.now_steps_+=1
        return self.min_lr_ + coeff*delta_lr



    

