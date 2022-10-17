import { Component } from '@angular/core';
import * as speechCommands from '@tensorflow-models/speech-commands';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  constructor() { }

  title = 'transfer-lernen';
  private recognizer: any;
  private examples: any = [];
  private model: any;
  private NUM_FRAMES = 3;
  private INPUT_SHAPE = [this.NUM_FRAMES, 232, 1];

  async ngOnInit() {
    this.app()
  }

  private predictWord() {
    // Array of words that the recognizer is trained to recognize.
    const words = this.recognizer.wordLabels();
    this.recognizer.listen((scores: any) => {
      // Turn scores into a list of (score,word) pairs.
      scores = Array.from(scores).map((s, i) => ({ score: s, word: words[i] }));
      // Find the most probable word.
      scores.sort((s1: any, s2: any) => s2.score - s1.score);
      const console = document.querySelector('#console');
      console!.textContent = scores[0].word;

    }, { probabilityThreshold: 0.75 });
  }

  private async app() {
    this.recognizer = speechCommands.create('BROWSER_FFT');
    await this.recognizer.ensureModelLoaded();
    this.buildModel();
  }

  // One frame is ~23ms of audio.

  public collect(label: any) {
    if (this.recognizer.isListening()) {
      return this.recognizer.stopListening();
    }
    if (label == null) {
      return;
    }
    this.recognizer.listen(async (spectrogramParam: { spectrogram: Spectrogram }) => {
      let vals = this.normalize(spectrogramParam.spectrogram.data.subarray(-spectrogramParam.spectrogram.frameSize * this.NUM_FRAMES));
      this.examples.push({ vals, label });
      const console = document.querySelector('#console');
      console!.textContent =
        `${this.examples.length} examples collected`;
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
  }

  private normalize(x: any) {
    const mean = -100;
    const std = 10;
    return x.map((x: any) => (x - mean) / std);
  }



  public async train() {
    this.toggleButtons(false);
    const ys = tf.oneHot(this.examples.map((e: any) => e.label), 3);
    const xsShape = [this.examples.length, ...this.INPUT_SHAPE];
    const xs = tf.tensor(this.flatten(this.examples.map((e: any) => e.vals)), xsShape);

    await this.model.fit(xs, ys, {
      batchSize: 16,
      epochs: 10,
      callbacks: {
        onEpochEnd: (epoch: any, logs: any) => {
          const console = document.querySelector('#console');
          console!.textContent =
            `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
        }
      }
    });
    tf.dispose([xs, ys]);
    this.toggleButtons(true);
  }

  private buildModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.depthwiseConv2d({
      depthMultiplier: 8,
      kernelSize: [this.NUM_FRAMES, 3],
      activation: 'relu',
      inputShape: this.INPUT_SHAPE
    }));
    this.model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
    this.model.add(tf.layers.flatten());
    this.model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    const optimizer = tf.train.adam(0.01);
    this.model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }

  private toggleButtons(enable: any) {
    document.querySelectorAll('button').forEach(b => b.disabled = !enable);
  }

  private flatten(tensors: any) {
    const size = tensors[0].length;
    const result = new Float32Array(tensors.length * size);
    tensors.forEach((arr: any, i: any) => result.set(arr, i * size));
    return result;
  }

  private async moveSlider(labelTensor: any) {
    const label = (await labelTensor.data())[0];
    const console = document.getElementById('console');
    console!.textContent = label;
    if (label == 2) {
      return;
    }
    let delta = 0.1;
    const slider = document.getElementById('output') as any;
    let prevValue = slider!.valueAsNumber;
    prevValue =
      prevValue + (label === 0 ? -delta : delta);
    slider.valueAsNumber = prevValue;
  }

  public listen() {
    if (this.recognizer.isListening()) {
      this.recognizer.stopListening();
      this.toggleButtons(true);
      const listen = document.getElementById('listen');
      listen!.textContent = 'Listen';
      return;
    }
    this.toggleButtons(false);
    const listen = document.getElementById('listen') as any;
    listen!.textContent = 'Stop';
    listen!.disabled = false;

    this.recognizer.listen(async (spectrogramParam: { spectrogram: Spectrogram }) => {
      const vals = this.normalize(spectrogramParam.spectrogram.data.subarray(-spectrogramParam.spectrogram.frameSize * this.NUM_FRAMES));
      const input = tf.tensor(vals, [1, ...this.INPUT_SHAPE]);
      const probs = this.model.predict(input);
      const predLabel = probs.argMax(1);
      await this.moveSlider(predLabel);
      tf.dispose([input, probs, predLabel]);
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
  }
}

export class Spectrogram {
  frameSize: any;
  data: any;
}