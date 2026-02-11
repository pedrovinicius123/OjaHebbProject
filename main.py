from oja.oja_rl import OjaRL
from utils.manager import EnvTrainer
from environ import test, record_videos
import gymnasium as gym

if __name__ == '__main__':

    trainer = EnvTrainer(n_features=14)
    model = trainer.run(5)
    
    test(model)
    record_videos(model)
