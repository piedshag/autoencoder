use ndarray::Array;
use ndarray_rand::{
    rand::{self, Rng},
    rand_distr::{Normal, Uniform},
    RandomExt,
};
use std::fmt::Display;

trait Layer {
    fn forward(&mut self, input: &Array<f64, ndarray::Ix2>) -> Array<f64, ndarray::Ix2>;
    fn backward(&mut self, input: &Array<f64, ndarray::Ix2>) -> Array<f64, ndarray::Ix2>;
}

struct Linear {
    weights: Array<f64, ndarray::Ix2>,
    bias: Array<f64, ndarray::Ix1>,
    inputs: Option<Array<f64, ndarray::Ix2>>,
}

impl Display for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.weights)
    }
}

impl Linear {
    fn new(input_size: usize, output_size: usize) -> Self {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let weights = Array::<f64, _>::random((input_size, output_size), dist);
        let bias = Array::<f64, _>::random(output_size, dist);
        Self {
            inputs: None,
            weights,
            bias,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Array<f64, ndarray::Ix2>) -> Array<f64, ndarray::Ix2> {
        self.inputs = Some(input.clone());
        let out = sigmoid(input.dot(&self.weights) + &self.bias);
        out
    }

    fn backward(&mut self, dldz: &Array<f64, ndarray::Ix2>) -> Array<f64, ndarray::Ix2> {
        // dL/dw = dL/dz * dz/dw
        // z = sigmoid(x*w + b)
        // dz/dw = sigmoid(x*w + b) * (1 - sigmoid(x*w + b)) * x

        let sigmoid_derivative = sigmoid(
            self.inputs
                .as_ref()
                .expect("expected inputs")
                .dot(&self.weights)
                + &self.bias,
        );

        let local_grad = dldz * &sigmoid_derivative;
        let dldw = self.inputs.as_ref().unwrap().t().dot(&local_grad);
        let bias_gradient = local_grad.sum_axis(ndarray::Axis(0));
        let input_gradient = local_grad.dot(&self.weights.t());

        self.weights = &self.weights - &dldw * 0.001;
        self.bias = &self.bias - &bias_gradient * 0.001;

        input_gradient
    }
}

struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    fn forward(&mut self, input: &Array<f64, ndarray::Ix2>) -> Array<f64, ndarray::Ix2> {
        self.layers
            .iter_mut()
            .fold(input.clone(), |input, layer| layer.forward(&input))
    }

    fn backward(&mut self, loss_gradient: &Array<f64, ndarray::Ix2>) {
        self.layers
            .iter_mut()
            .rev()
            .fold(loss_gradient.clone(), |input, layer| layer.backward(&input));
    }
}

fn softmax(x: Array<f64, ndarray::Ix2>) -> Array<f64, ndarray::Ix2> {
    let max = x.iter().fold(f64::MIN, |m, &val| m.max(val));
    let x_exp: Array<f64, _> = x.mapv(|val| (val - max).exp());
    &x_exp / x_exp.sum()
}

fn mean_squared(x: &Array<f64, ndarray::Ix2>, y: &Array<f64, ndarray::Ix2>) -> f64 {
    let diff = x - y;
    (&diff * &diff).sum() / (x.len() as f64)
}

fn sigmoid(x: Array<f64, ndarray::Ix2>) -> Array<f64, ndarray::Ix2> {
    1.0 / (1.0 + (-x).mapv(|val| val.exp()))
}

fn main() {
    let epochs = 10000;
    let batch_size = 1;
    let output_size = 8;

    let input = Array::random((batch_size, 100), Uniform::new(0.0, 1.0));
    let labels = input.clone();

    // let labels = Array::zeros((batch_size, output_size));
    // let rng = rand::thread_rng().gen_range(0..output_size);
    // labels[[0, rng]] = 1.0;

    let layer1 = Box::new(Linear::new(100, 20));
    let layer2 = Box::new(Linear::new(20, 5));
    let layer3 = Box::new(Linear::new(5, 20));
    let layer4 = Box::new(Linear::new(20, 100));

    let mut network = Network::new(vec![layer1, layer2, layer3, layer4]);

    for i in 0..epochs {
        let output = network.forward(&input);
        let loss = mean_squared(&output, &labels);

        println!("epoch: {} loss: {}", i, loss);

        if i == epochs - 1 {
            println!("output: {}", output);
            println!("labels: {}", labels);
        }

        let output_gradient = output - &labels; // for cross-entropy loss with softmax output
        network.backward(&output_gradient);
    }
}
