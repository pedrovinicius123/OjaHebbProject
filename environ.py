import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random, time, os

def test(model):
    # Parâmetros do treinamento
    EPISODES = 100  # Número de episódios para executar
    MAX_STEPS = 500  # Número máximo de passos por episódio

    # Inicializar ambiente
    env = gym.make('CartPole-v1')

    # Lista para armazenar scores
    scores = []
    episode_numbers = []

    print("Iniciando execução do CartPole-v1...")
    print(f"Total de episódios: {EPISODES}\n")

    if not os.path.exists('results'):
        os.mkdir('results')

    n_files = len(os.listdir('results'))

    if not os.path.exists(f'plots/fig-{n_files}'):
        os.makedirs(f'plots/fig-{n_files}')

    # Executar episódios
    for episode in range(EPISODES):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state  # Para compatibilidade com novas versões do Gym   
        reward = 1
        total_reward = 1

        for step in range(MAX_STEPS):
            # Ação aleatória (para demonstração
            state = state.reshape(1, state.size)
            action = model(state, total_reward)

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

    # Calcular estatísticas
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)

    with open(f'results/result-{n_files}.log', 'a') as file:
        file.write("\n" + "="*50 + "\n")
        file.write("ESTATÍSTICAS DOS SCORES:\n") 
        file.write(f"Média: {mean_score:.2f}\n")
        file.write(f"Mediana: {median_score:.2f}\n")
        file.write(f"Melhor score: {max_score}\n")
        file.write(f"Pior score: {min_score}\n")
        file.write("="*50)

    #print(scores)

    # Criar gráfico
    plt.figure(figsize=(12, 6))

    # Gráfico de linhas dos scores
    plt.subplot(1, 2, 1)
    plt.plot(episode_numbers, scores, 'b-', linewidth=1.5, alpha=0.7)
    plt.scatter(episode_numbers, scores, color='blue', s=30, alpha=0.6)
    plt.axhline(y=mean_score, color='r', linestyle='--', label=f'Média: {mean_score:.1f}')
    plt.axhline(y=median_score, color='g', linestyle='--', label=f'Mediana: {median_score:.1f}')

    plt.xlabel('Número do Episódio', fontsize=12)
    plt.ylabel('Score (Recompensa Total)', fontsize=12)
    plt.title('Scores por Episódio no CartPole-v1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Histograma da distribuição dos scores
    plt.subplot(1, 2, 2)
    n, bins, patches = plt.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(x=mean_score, color='r', linestyle='--', linewidth=2, label=f'Média: {mean_score:.1f}')
    plt.axvline(x=median_score, color='g', linestyle='--', linewidth=2, label=f'Mediana: {median_score:.1f}')

    plt.xlabel('Score (Recompensa Total)', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.title('Distribuição dos Scores', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.suptitle(f'Análise de Desempenho - CartPole-v1 ({EPISODES} episódios)', 
             fontsize=16, fontweight='bold', y=1.02)

    

    plt.savefig(f'plots/fig-{n_files}/stats.png')

    # Versão alternativa simplificada (apenas gráfico de linha)
    plt.figure(figsize=(10, 5))
    plt.plot(episode_numbers, scores, 'b-', linewidth=2, alpha=0.8, marker='o', markersize=4)
    plt.fill_between(episode_numbers, scores, alpha=0.2, color='blue')

    # Linha da média móvel (média dos últimos 10 episódios)
    window_size = 10
    if len(scores) >= window_size:
        moving_avg = []
        for i in range(len(scores)):
            if i < window_size:
                moving_avg.append(np.mean(scores[:i+1]))
            else:
                moving_avg.append(np.mean(scores[i-window_size+1:i+1]))
    
        plt.plot(episode_numbers, moving_avg, 'r-', linewidth=2, 
             label=f'Média Móvel ({window_size} episódios)')

    plt.xlabel('Episódio', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Desempenho no CartPole-v1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/fig-{n_files}/median.png')

def record_videos(model):

    # Configuration
    num_eval_episodes = 4
    env_name = "CartPole-v1"  # Replace with your environment

    # Create environment with recording capabilities
    env = gym.make(env_name, render_mode="rgb_array")  # rgb_array needed for video recording

    # Add video recording for every episode
    env = RecordVideo(
       env,
       video_folder="cartpole-agent",    # Folder to save videos
       name_prefix="eval",               # Prefix for video filenames
       episode_trigger=lambda x: True    # Record every episode
    )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    print(f"Starting evaluation for {num_eval_episodes} episodes...")
    print(f"Videos will be saved to: cartpole-agent/")

    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0

        episode_over = False
        while not episode_over:
            # Replace this with your trained agent's policy
            action = model(obs, 1)  # Random policy for demonstration

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            episode_over = terminated or truncated

        print(f"Episode {episode_num + 1}: {step_count} steps, reward = {episode_reward}")


    env.close()

    # Print summary statistics
    print(f'\nEvaluation Summary:')
    print(f'Episode durations: {list(env.time_queue)}')
    print(f'Episode rewards: {list(env.return_queue)}')
    print(f'Episode lengths: {list(env.length_queue)}')

    # Calculate some useful metrics
    avg_reward = np.sum(env.return_queue)
    avg_length = np.sum(env.length_queue)
    std_reward = np.std(env.return_queue)

    print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
    print(f'Average episode length: {avg_length:.1f} steps')
    print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')
