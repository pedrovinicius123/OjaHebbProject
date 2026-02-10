import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random, time

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

    # Executar episódios
    for episode in range(EPISODES):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state  # Para compatibilidade com novas versões do Gym
        total_reward = 1
        reward = 1
    

        for step in range(MAX_STEPS):
            # Ação aleatória (para demonstração)
            action = model.forward_learn(state, total_reward)
        
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
                if total_reward < 25:
                    reward = -3

                break
    
        # Registrar score
        scores.append(total_reward)
        episode_numbers.append(episode + 1)
    
        # Exibir progresso a cada 10 episódios
        if (episode + 1) % 1 == 0:
            print(f"Episódio {episode + 1}/{EPISODES}, Score: {total_reward}")

    # Fechar ambiente
    env.close()

    # Calcular estatísticas
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)

    print("\n" + "="*50)
    print("ESTATÍSTICAS DOS SCORES:") 
    print(f"Média: {mean_score:.2f}")
    print(f"Mediana: {median_score:.2f}")
    print(f"Melhor score: {max_score}")
    print(f"Pior score: {min_score}")
    print("="*50)

    print(scores)

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
    plt.show()

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
    plt.title('Desempenho no CartPole-v1 (Ações Aleatórias)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()