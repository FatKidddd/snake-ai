import React from 'react';
import ReactDOM from 'react-dom/client';
import reportWebVitals from './reportWebVitals';
import * as tf from '@tensorflow/tfjs';
import Sketch from 'react-p5';
import p5Types from 'p5';
import DQNAgent from './dqn';

class Game {
  p5: p5Types;
  w: number;
  h: number;
  blockSize: number;
  pos: p5Types.Vector[];
  vel: p5Types.Vector;
  hunger: number;
  food: p5Types.Vector;
  
  constructor(p5: p5Types, w: number, h: number, blockSize: number) {
    this.p5 = p5;
    this.w = w;
    this.h = h;
    this.blockSize = blockSize;
    this.pos = [this.p5.createVector(Math.floor(this.w / 2), Math.floor(this.h / 2))];
    this.vel = this.p5.createVector(1, 0);
    this.hunger = this.pos.length * 50;
    this.food = this.spawnFood();
  }

  update(action: number): [number, boolean] {
    action--; // [turn left, straight, turn right]
    const vels = [[0, -1], [1, 0], [0, 1], [-1, 0]];
    const velIdx = vels.findIndex(e => e[0] === this.vel.x && e[1] === this.vel.y);
    const dir = (velIdx+action+4)%4;
    this.vel = this.p5.createVector(...vels[dir]);
    for (let i=this.pos.length-1; i>0; i--) {
      this.pos[i] = this.pos[i-1];
    }
    const newVel = this.pos[0].copy().add(this.vel);
    this.pos[0] = newVel;

    const done = this.gameOver();
    if (done) return [-10, done];

    this.hunger--;

    let reward = 0;
    if (this.pos[0].equals(this.food)) {
      this.food = this.spawnFood();
      this.pos.push(this.pos[this.pos.length-1]);
      this.hunger = this.pos.length * 50;
      reward = 10;
    }
		return [reward, done];
  }

  spawnFood() {
    while (true) {
      const genPos = this.p5.createVector(
        this.p5.floor(this.p5.random() * this.w),
        this.p5.floor(this.p5.random() * this.h)
      );
      let ok = true;
      for (const pos of this.pos) {
        if (pos.equals(genPos)) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      return genPos;
    }
  }

  gameOver() {
    // check collision
    const head = this.pos[0];
    if (!(0 <= head.x && head.x < this.w) || !(0 <= head.y && head.y < this.h))
      return true;

    for (let i=1; i<this.pos.length; i++)
      if (this.pos[i].equals(this.pos[0])) 
        return true;

    // check hunger
    return this.hunger === 0;
  }

  getGrid() {
    const grid: number[][] = new Array(this.h).map(e => new Array(this.w).fill(0));
    for (const pos of this.pos) {
      grid[pos.x][pos.y] = 1;
    }
    grid[this.food.x][this.food.y] = 3;
    grid[this.pos[0].x][this.pos[0].y] = 2;
    return grid;
  }

  getState() {
    const vels = [[0, -1], [1, 0], [0, 1], [-1, 0]];
    const dir = vels.findIndex(e => e[0] === this.vel.x && e[1] === this.vel.y);
    const oneHotVelDir: number[] = new Array(4).fill(0);
    oneHotVelDir[dir] = 1;

    // distance from walls
    const hx = this.pos[0].x, hy = this.pos[0].y;
    let top = hy;
    let bot = this.h - hy;
    let left = hx;
    let right = this.w - hx;
		let dtr = this.p5.min(top, right);
    let dbr = this.p5.min(bot, right);
    let dbl = this.p5.min(bot, left);
    let dtl = this.p5.min(top, left);

    // distance from snake body
    for (let i=1; i<this.pos.length; i++) {
      const x = this.pos[i].x;
      const y = this.pos[i].y;
      
			const dx = x-hx, dy = y-hy;
			if (dy > 0) bot = this.p5.min(bot, dy);
			if (dy < 0) top = this.p5.min(top, -dy);
			if (dx < 0) left = this.p5.min(left, -dx);
			if (dx > 0) right = this.p5.min(right, dx);
			if (this.p5.abs(dx)==this.p5.abs(dy)) {
        if (dy < 0 && dx > 0) dtr = this.p5.min(dtr, this.p5.abs(dx));
				if (dy > 0 && dx > 0) dbr = this.p5.min(dbr, this.p5.abs(dx));
				if (dy > 0 && dx < 0) dbl = this.p5.min(dbl, this.p5.abs(dx));
				if (dy < 0 && dx < 0) dtl = this.p5.min(dtl, this.p5.abs(dx));
      }
    }
    
    const [u, r, d, l] = new Array(4).fill(0).map((_, idx) => idx === dir);
    const state = [
      Number(
        (u && this.isCollision(this.p5.createVector(...vels[0]))) ||
        (r && this.isCollision(this.p5.createVector(...vels[1]))) ||
        (d && this.isCollision(this.p5.createVector(...vels[2]))) ||
        (l && this.isCollision(this.p5.createVector(...vels[3])))
      ),
			
      //left
      Number(
        (r && this.isCollision(this.p5.createVector(...vels[0]))) ||
        (d && this.isCollision(this.p5.createVector(...vels[1]))) ||
        (l && this.isCollision(this.p5.createVector(...vels[2]))) ||
        (u && this.isCollision(this.p5.createVector(...vels[3])))
      ),

      //right
      Number(
        (l && this.isCollision(this.p5.createVector(...vels[0]))) ||
        (u && this.isCollision(this.p5.createVector(...vels[1]))) ||
        (r && this.isCollision(this.p5.createVector(...vels[2]))) ||
        (d && this.isCollision(this.p5.createVector(...vels[3])))
      )
    ];
    // add walls dist relative to snake head
		// const state = [top, dtr, right, dbr, bot, dbl, left, dtl]
    // const state = [top, right, bot, left];
    // const state = temp.slice(0);
    // console.log(state)
    // const n = temp.length;
    // for (let i=0; i<n; i++) {
    //   state[i] = temp[(i+dir)%n]; // top will be snake dir
    // }

    // add vel dir
		state.push(...oneHotVelDir);

    // add food dir relative to snake head
    const fx = this.food.x, fy = this.food.y;
		const dfx = fx-hx, dfy = fy-hy;
		const foodDir = [Number(dfy<0), Number(dfx>0), Number(dfy>0), Number(dfx<0)];
    // const rotatedFoodDir = foodDir.slice();
    // const m = foodDir.length;
    // for (let i=0; i<m; i++) {
    //   rotatedFoodDir[i] = foodDir[(i+dir)%m]; // top will be snake dir
    // }
		state.push(...foodDir);
    
    
		return state;
  }

  isCollision(vel: p5Types.Vector) {
    const futurePos = this.pos[0].copy().add(vel);
    if (!(0<=futurePos.x && futurePos.x<this.w) || !(0<=futurePos.y && futurePos.y<this.h))
      return true;
    for (let i=1; i<this.pos.length; i++)
      if (this.pos[i].equals(futurePos))
        return true;
    return false;
  }

  display() {
    this.p5.background(0);

    for (const pos of this.pos) {
      const x = pos.x, y = pos.y;
      this.p5.fill(255);
      this.p5.noStroke();
      this.p5.rect(x*this.blockSize, y*this.blockSize, this.blockSize, this.blockSize);
    }

    this.p5.fill(255, 0, 0);
    this.p5.rect(this.food.x*this.blockSize, this.food.y*this.blockSize, this.blockSize, this.blockSize);
  }
}

const width = 10;
const height = 10;
const blockSize = 20;
let game: Game;

const stateSize = 11;
const actionSize = 3;
const batchSize = 1000;
const dqn = new DQNAgent(stateSize, actionSize, batchSize);
//dqn.load("weights4.h5")
let episode = 0;
let scores: number[] = [];

let globalDone = false;
const currentGameMemory: any[] = [];

interface ComponentProps {
	//Your component props
}

const App: React.FC<ComponentProps> = (props: ComponentProps) => {
	const setup = (p5: p5Types, canvasParentRef: Element) => {
		p5.createCanvas(width*blockSize, height*blockSize).parent(canvasParentRef);
    game = new Game(p5, width, height, blockSize);
	};

	const draw = async (p5: p5Types) => {
    if (!globalDone) {
      game.display();
      const state = game.getState();
      const tensorState = tf.tensor2d(state, [1, state.length]);
      const action = await dqn.get_action(tensorState);
      const argMaxedAction = tf.argMax(action).dataSync()[0];
      const [reward, done] = game.update(argMaxedAction);
      const nextState = game.getState();
    
      // train
      // convert everything to javascript array first and with the correct dims, commented shape
      dqn.remember(state, [action], [reward], nextState, [done]);
      globalDone = true;
      // await dqn.train_short_memory([state], [action], [reward], [nextState], [done]);
      if (done) {
        const loss = await dqn.train_long_memory();
        ++episode;
  			scores.push(game.pos.length);
  			console.log('Episode ' + episode + ' - Length: ' + game.pos.length + ', Loss: ' + loss);
        game = new Game(p5, width, height, blockSize);
        //dqn.save("weights4.h5")
      }
      globalDone = false;
    }
	};
	return <Sketch setup={setup} draw={draw} />;
};

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(<App />);

reportWebVitals();