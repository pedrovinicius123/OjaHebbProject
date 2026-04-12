import gymnasium as gym
import os

def test_moonlander(model):
    # Parâmetros do treinamento
    EPISODES = 100  # Número de episódios para executar
    MAX_STEPS = 500  # Número máximo de passos por episódio

    # Inicializar ambiente
    env = gym.make('LunarLander-v3')

    # Lista para armazenar scores
    scores = []
    episode_numbers = []

    print("Iniciando execução do LunarLander-v3...")
    print(f"Total de episódios: {EPISODES}\n")

    if not os.path.exists('results'):
        os.mkdir('results')

    n_files = len(os.listdir('results'))

    if not os.path.exists(f'plots-lunar_lander/fig-{n_files}'):
        os.makedirs(f'plots-lunar_lander/fig-{n_files}')

    # Executar episódios
    for episode in range(EPISODES):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state  # Para compatibilidade com novas versões do Gym   
        reward = 1
        total_reward = 1

        for step in range(MAX_STEPS):
            # Ação aleatória (para demonstração
            state = state.reshape(1, state.size)
            action = model(state, 1, intervals=[(-float("inf"), 0), (0, 1), (1, 2), (2, 3), (3, float("inf"))])
            print(action)
            # Executar ação
            result = env.step(action)
        
            # Verificar formato da resposta (novas vs antigas versões do Gym)
            if len(result) == 5:  # Nova versão do Gym (0.26.0+)
                state, reward, terminated, truncated, info = result
                done = terminated
            else:  # Versões mais antigas
                state, reward, done, info = result
        
            total_reward += reward
        
            if done:
                break
    
        # Registrar score
        
        scores.append(total_reward)
        episode_numbers.append(episode + 1)
    
        # Exibir progresso a cada 10 episódios
        if (episode + 1) % 1 == 0:
            
            episode_reward = f"Episódio {episode + 1}/{EPISODES}, Score: {total_reward}\n"
            
            with open(f'results/result-{n_files}.log', 'a') as file:
                file.write(episode_reward)

        total_reward = 1
    # Fechar ambiente
    env.close()
