import os
import time
from .flow import _save_ckpt
import numpy as np
import tensorflow as tf
import pickle

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)

def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    feed_dict = self.framework.shuffle()
    loss_op = self.framework.loss

    i = 0
    while True:
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        fetches = [self.train_op, loss_op] 

        if self.FLAGS.summary:
            fetches.append(self.summary_op)
        try:
            fetched = self.sess.run(fetches, feed_dict)
        except tf.errors.OutOfRangeError:
            break
            
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)
        i+=1

    if ckpt: _save_ckpt(self, *args)
        

