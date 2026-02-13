from oja.oja_rl import OjaRL
from utils.manager import EnvTrainer
from environ import test, record_videos
import gymnasium as gym

if __name__ == '__main__':
    n_logs = 20
    trainer = EnvTrainer(n_features=14)
    for log in range(n_logs):
       
       model = trainer.run(5, log)
       model.learning_rate = .00002
       test(model)
    
    record_videos(model)
