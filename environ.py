import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from matplotlib.lines import Line2D


def test_env_peaks_with_logging(model, env):
    """
    Testa o modelo no ambiente com logging detalhado de cada passo.
    Retorna um DataFrame com o histórico completo e plota análises específicas.
    """
    # Reset do ambiente
    reward = 0.0
    tot = 0
    frozen = False

    # LISTA PRINCIPAL para armazenar o histórico de CADA PASSO
    step_history = []
    # Lista auxiliar para recompensas por episódio
    episode_rewards = []
    episode_peaks = []

    # ========== CONFIGURAÇÃO DO LOGGING ==========
    num_episodes_for_logging = 10

    for episode in range(num_episodes_for_logging):
        s = random.randint(0, 100)
        observation, info = env.reset(seed=s)
        frozen = False
        tot = 0
        episode_step_history = []

        # Número de passos por episódio
        for step in range(1000):
            # 1. REGISTRA O ESTADO ATUAL ANTES DA AÇÃO
            log_entry = {
                'episode': episode,
                'step': step,
                'frozen': frozen,
                'tot_before_action': tot,
                'Q_sa_before': model.Q_sa.copy().tolist(),  # Converte para lista
                'probs_before': model.probs.copy().tolist(),
                'reward_before': float(reward)
            }

            # 2. ESCOLHE E EXECUTA A AÇÃO
            action = model.forward_learn(observation, reward)
            learning_phase = 'active_learning'

            log_entry['action'] = int(action)
            log_entry['learning_phase'] = learning_phase

            # 3. PASSA NO AMBIENTE
            observation, reward, terminated, truncated, _ = env.step(action)
            log_entry['reward_after'] = float(reward)
            log_entry['terminated'] = bool(terminated)
            log_entry['truncated'] = bool(truncated)

            # 4. ATUALIZA ESTADOS APÓS O PASSO
            tot += 1
            log_entry['tot_after'] = tot
            log_entry['Q_sa_after'] = model.Q_sa.copy().tolist()
            log_entry['probs_after'] = model.probs.copy().tolist()

            # 5. ARMAZENA O REGISTRO
            episode_step_history.append(log_entry)

            # 6. VERIFICA FIM DO EPISÓDIO
            if terminated or truncated:
                episode_rewards.append(tot)
                reward = -1

                # Lógica de congelamento
                if tot > 60:
                    episode_peaks.append(tot)
                    reward = 2
                    frozen = True
                    print(f"🚀 EPISÓDIO {episode} TERMINOU COM PICO! Recompensa={tot}")
                else:
                    frozen = not frozen

                # SALVA histórico do episódio
                step_history.extend(episode_step_history)
                break

        else:
            # Caso o episódio não termine naturalmente
            episode_rewards.append(tot)
            step_history.extend(episode_step_history)
            frozen = not frozen

    # ========== CONVERSÃO PARA DATAFRAME ==========
    df = pd.DataFrame(step_history)
    print(f"\n✅ Logging completo! Capturados {len(df)} passos individuais.")
    print(f"📋 Estrutura do DataFrame: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # ========== PREPARAÇÃO PARA PLOTAGEM ==========
    # Extrai dados para plotagem (CORREÇÃO DO PROBLEMA)
    num_actions = model.Q_sa.shape[1]

    # Prepara arrays para valores Q e probabilidades por ação
    q_values_by_action = {f'Q_a{i}': [] for i in range(num_actions)}
    probs_by_action = {f'P_a{i}': [] for i in range(num_actions)}

    for _, row in df.iterrows():
        for i in range(num_actions):
            q_values_by_action[f'Q_a{i}'].append(row['Q_sa_after'][i])
            probs_by_action[f'P_a{i}'].append(row['probs_after'][i][0])

    # ========== VISUALIZAÇÕES ESPECÍFICAS ==========
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Análise Microscópica da Física Interna do OjaRL', fontsize=16, fontweight='bold')

    # 3. PROBABILIDADES DAS AÇÕES (CORRIGIDO)
    colors = ['blue', 'red', 'green', 'purple'][:num_actions]
    for i in range(num_actions):
        axs[1, 0].plot(df['step'], probs_by_action[f'P_a{i}'],
                       color=colors[i], label=f'Ação {i}', alpha=0.7, linewidth=1.5)
    axs[1, 0].set_title('Evolução das Probabilidades de Ação (probs)')
    axs[1, 0].set_xlabel('Passo Global')
    axs[1, 0].set_ylabel('Probabilidade')
    axs[1, 0].legend(loc='best')
    axs[1, 0].grid(True, alpha=0.3)

    # 4. VALORES Q (Q_sa) PARA CADA AÇÃO (CORRIGIDO)
    for i in range(num_actions):
        axs[1, 1].plot(df['step'], q_values_by_action[f'Q_a{i}'],
                       color=colors[i], label=f'Q[a{i}]', alpha=0.7, linewidth=1.5)
    axs[1, 1].set_title('Evolução dos Valores Q (Q_sa)')
    axs[1, 1].set_xlabel('Passo Global')
    axs[1, 1].set_ylabel('Valor Q')
    axs[1, 1].legend(loc='best')
    axs[1, 1].grid(True, alpha=0.3)

    # 5. RECOMPENSAS POR PASSO E FASE DE APRENDIZADO
    color_map = {'active_learning': 'blue', 'frozen_execution': 'orange'}
    point_colors = [color_map[phase] for phase in df['learning_phase']]
    axs[2, 0].scatter(df['step'], df['reward_after'], c=point_colors, alpha=0.5, s=10)
    axs[2, 0].set_title('Recompensas por Passo (Cor por Fase)')
    axs[2, 0].set_xlabel('Passo Global')
    axs[2, 0].set_ylabel('Recompensa Imediata')
    # Legenda manual
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Active Learning',
               markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Frozen Execution',
               markerfacecolor='orange', markersize=8)
    ]
    axs[2, 0].legend(handles=legend_elements, loc='best')
    axs[2, 0].grid(True, alpha=0.3)

    # 6. GRÁFICO TRADICIONAL DE RECOMPENSAS POR EPISÓDIO
    axs[2, 1].plot(range(len(episode_rewards)), episode_rewards, 'b-', marker='o', markersize=4)
    axs[2, 1].set_title('Recompensa Total por Episódio')
    axs[2, 1].set_xlabel('Número do Episódio')
    axs[2, 1].set_ylabel('Recompensa Total (TOT)')
    axs[2, 1].grid(True, alpha=0.3)
    # Destacar episódios com picos
    for i, rew in enumerate(episode_rewards):
        if rew > 60:
            axs[2, 1].plot(i, rew, 'ro', markersize=8)
            axs[2, 1].text(i, rew + 5, f'{rew}', ha='center', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.show()

    env.close()

    # ========== ANÁLISE DE DADOS ADICIONAL ==========
    print("\n" + "=" * 60)
    print("ANÁLISE DETALHADA DOS DADOS CAPTURADOS")
    print("=" * 60)

    if len(episode_rewards) > 0:
        print(f"📊 Estatísticas das {len(episode_rewards)} execuções:")
        print(f"   Recompensa média: {np.mean(episode_rewards):.1f}")
        print(f"   Recompensa máxima: {np.max(episode_rewards)}")
        print(f"   Recompensa mínima: {np.min(episode_rewards)}")
        print(f"   Desvio padrão: {np.std(episode_rewards):.1f}")

        if episode_peaks:
            print(f"\n🎯 Estatísticas dos {len(episode_peaks)} picos (>60):")
            print(f"   Média dos picos: {np.mean(episode_peaks):.1f}")
            print(f"   Melhor pico: {np.max(episode_peaks)}")

            # Análise do VS nos picos
            peak_episodes = [i for i, rew in enumerate(episode_rewards) if rew > 60]

    return df, episode_rewards, episode_peaks