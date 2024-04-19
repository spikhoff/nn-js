// Lae TensorFlow.js
import * as tf from '@tensorflow/tfjs';

// Loo mudel: järjestikuline mudel on lihtne mudel, kus sisendid liiguvad ühest kihist teise
const model = tf.sequential();

// Lisa esimene kiht (peidetud kiht)
// dense on tihedalt seotud kiht, units on neuronite arv, inputShape on sisendi kuju
model.add(tf.layers.dense({units: 4, inputShape: [2], activation: 'sigmoid'}));

// Lisa väljundkiht
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

// Kompilaator: määra optimiseerija, kaotusfunktsioon ja metrika
model.compile({
  optimizer: 'sgd',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
});

// Loo mõned andmed koolitamiseks
const xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
const ys = tf.tensor2d([[0], [1], [1], [0]]);

// Koolita mudel
async function train() {
  const response = await model.fit(xs, ys, {
    epochs: 500, // kui mitu korda koolitusandmeid kasutatakse
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}`);
      }
    }
  });
  console.log('Training Complete');
}

// Kutsu koolitamise funktsioon
train().then(() => {
  // Teha ennustused
  const output = model.predict(tf.tensor2d([[0, 1], [1, 1], [1, 0]]));
  output.print(); // Print the output of the predictions
});
