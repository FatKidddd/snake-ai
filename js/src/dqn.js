import * as tf from '@tensorflow/tfjs';

export default class DQNAgent {
  constructor(state_size, action_size, batch_size=32) {
    this.state_size = state_size;
    this.action_size = action_size;
    this.memory = [];
    this.memory_max_len = 10000;
    this.gamma = 0.95;
    this.epsilon = 1.0;
    this.epsilon_min = 0.001;
    this.epsilon_decay = 0.995;
    this.learning_rate = 0.001;
    this.batch_size = batch_size;
    this.model = this.create_model();
  }
	
  create_model() {
    const m = tf.sequential();
    m.add(tf.layers.dense({ inputShape: [this.state_size], units: 128, activation: 'relu' }));
    m.add(tf.layers.dense({ units: this.action_size, activation: 'softmax' }));

    m.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
    return m;
  }

  remember(state, action, reward, next_state, done) {
    this.memory.push([state, action, reward, next_state, done]);
    if (this.memory.length > this.memory_max_len) this.memory.shift();
  }

  async get_action(state) {
    // console.log('got action')
    const randVal = await tf.randomUniform([1]).arraySync()[0];
    if (randVal <= this.epsilon) {
      if (this.epsilon > this.epsilon_min) {
        this.epsilon *= this.epsilon_decay
      }
      const idx = await tf.randomUniform([1], 0, this.action_size, 'int32').arraySync()[0];
      const act_values = tf.oneHot(idx, this.action_size);
      // console.log(act_values)
      return await act_values.arraySync();
    }
    return await this.model.predict(state).arraySync()[0];
  }

  async train_step(states, actions, rewards, next_states, dones) {
    const targets = await this.model.predict(states).arraySync();

    const arrayActions = await actions.arraySync();
    const arrayDones = await dones.arraySync();
    const arrayRewards = await rewards.arraySync();
    const arrayNextStates = await next_states.arraySync();

    // console.log(arrayDones);
    for (let i = 0; i < arrayDones.length; i++) {
      // console.log('Reward: ', arrayRewards[i]);
      let q_new = arrayRewards[i];
      // console.log('Target: ', targets[i]);
      if (!arrayDones[i]) {
        const next_state = tf.expandDims(arrayNextStates[i], 0);
        const future_reward = await this.model.predict(next_state).arraySync()[0];
        // console.log('Future Reward: ', future_reward);
        // console.log('Max future reward: ', max(...future_reward));
        q_new = (arrayRewards[i] + this.gamma * Math.max(...future_reward));
        // console.log('Q new: ', q_new);
      }
      const action_idx = await tf.argMax(arrayActions[i]).dataSync()[0];
      // console.log(action_idx);
      targets[i][action_idx] = q_new;
      // console.log('Final Target: ', targets[i]);
    }

    const tensorTargets = tf.tensor(targets);

    const history = await this.model.fit(states, tensorTargets, { epochs: 1 });
    // console.log(history);
    return history.history.loss[0];
  }
	
  async train_short_memory(state, action, reward, next_state, done) {
    return await this.train_step(tf.tensor(state), tf.tensor(action), tf.tensor(reward), tf.tensor(next_state), tf.tensor(done));
  }

  async train_long_memory() {
    const minibatch = await randomSample(this.memory, Math.min(this.memory.length, this.batch_size));
    let [states, actions, rewards, next_states, dones] = zip(...minibatch);
    states = tf.tensor(states);
    actions = tf.squeeze(actions);
    rewards = tf.squeeze(rewards);
    next_states = tf.tensor(next_states);
    dones = tf.squeeze(dones);
    return await this.train_step(states, actions, rewards, next_states, dones);
  }
}

const zip = (...arr) => {
  return Array(Math.max(...arr.map(a => a.length))).fill().map((_, i) => arr.map(a => a[i]));
}

const randomSample = async (arr, sampleSize) => {
  let n = arr.length;
  const res = [];
  for (let i = 0; i < sampleSize; i++) {
    const idx = await tf.randomUniform([1], 0, n, 'int32').arraySync()[0];
    [arr[n - 1], arr[idx]] = [arr[idx], arr[n - 1]];
    n--;
    res.push(arr[idx]);
  }
  return res;
}