"""
@author: Daniel Shani
"""

from parameters import parameters
from run_model import train_agent

if __name__ == '__main__':
    if parameters.ego_q_learning:
        parameters.egamma = 0
        parameters.SR_lr_e = 0
    train_agent(parameters, save=True)
