import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve
from enviroment_csv import single_env
if __name__ == '__main__':
    #env = gym.make('CartPole-v0')
    env = single_env()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0001
    agent = Agent(n_actions=env.action_space, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space)
    n_games = 5000

    figure_file = 'plots/single_ppo.png'

    best_score = env.reward_origin
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        nextstep_count = 1
        while not done:
            nextstep_count = nextstep_count + 1
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action,nextstep_count)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action,
                                   prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        #print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
        #      'time_steps', n_steps, 'learning_steps', learn_iters)
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)