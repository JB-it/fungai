use std::f32::consts::E;

use ndarray::{Array2, Array1};
use rand::{rngs::ThreadRng, Rng};

#[derive(Clone)]
pub struct Layer {
    pub neurons: Array1<f32>,
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

pub enum CreationType {
    Randomized {amount: f32},
    Zeroes,
    Ones,
    Constants {number: f32}
}

#[derive(Clone)]
pub enum ActivationFunction {
    None,
    Sigmoid,
    Positive,
    ZeroOne,
}

impl Layer {
    pub fn new(input_size: usize, layer_size: usize, creation_type: &CreationType) -> Self {

        let mut t_neurons: Array1<f32> = Array1::zeros([layer_size]);
        let mut t_biases: Array1<f32>;
        let mut t_weights: Array2<f32>;

        match creation_type {
            CreationType::Randomized { amount } => {
                let mut rng = rand::thread_rng();

                t_biases = Array1::zeros([layer_size]);
                t_weights = Array2::zeros([layer_size, input_size]);
            },
            CreationType::Zeroes => {
                t_biases = Array1::zeros([layer_size]);
                t_weights = Array2::zeros([layer_size, input_size]);
            }
            CreationType::Ones => {
                t_biases = Array1::ones([layer_size]);
                t_weights = Array2::ones([layer_size, input_size]);
            }
            CreationType::Constants { number } => {
                t_biases = Array1::ones([layer_size]) * *number;
                t_weights = Array2::ones([layer_size, input_size]) * *number;
            }
        }

        Layer {
            neurons: t_neurons,
            weights: t_weights,
            biases: t_biases,
        }
    }

    pub fn calculate(&mut self, input: Array1<f32>, activation: &ActivationFunction) {
        self.neurons = self.weights.dot(&input);
        self.neurons += &self.biases;
        self.activate(activation);
    }

    pub fn activate(&mut self, activation: &ActivationFunction) {
        for n in self.neurons.iter_mut() {
            match activation {
                ActivationFunction::Sigmoid => {
                    *n = 1./(1.+E.powf(-*n));
                },
                ActivationFunction::Positive => {
                    *n = n.max(0.);
                },
                ActivationFunction::ZeroOne => {
                    *n = if *n < 0. {0.} else {1.};
                }
                _ => (),
            }
        }
    }

    pub fn get_input_size(&self) -> usize {
        self.weights.dim().1
    }

    pub fn get_output_size(&self) -> usize {
        self.neurons.dim()
    }

    pub fn mutate(&mut self, amount: f32, rng: &mut ThreadRng) {
        let mutate_weight = rng.gen_bool(0.5);

        if mutate_weight {
            //Mutate Weight
            let weight_x_index: usize = rng.gen::<usize>() % self.weights.dim().0 as usize;
            let weight_y_index: usize = rng.gen::<usize>() % self.weights.dim().1 as usize;
            let random_add: f32 = rng.gen::<f32>() % amount;

            let random_mut: f32 = match rng.gen_bool(0.5) {
                true => random_add,
                false => -random_add,
            };

            let weight = self.weights.get_mut([weight_x_index, weight_y_index]).unwrap();
            *weight += random_mut;

        } else {
            //Mutate Bias
            let bias_index: usize = rng.gen::<usize>() % self.biases.len() as usize;
            let random_add: f32 = rng.gen::<f32>() % amount;

            let random_mut: f32 = match rng.gen_bool(0.5) {
                true => random_add,
                false => -random_add,
            };

            self.biases[bias_index] += random_mut;
        }
    }

    pub fn print_layer(&self) {
        println!("Weights");
        println!("{}", self.weights);
        println!("Biases");
        println!("{}", self.biases);
        println!("Output");
        println!("{}", self.neurons);
    }
}

#[test]
fn new_layer() {
    let mut layer = Layer::new(3, 1, &CreationType::Zeroes);

    layer.calculate(Array1::zeros([3]), &ActivationFunction::None);

    assert_eq!(layer.neurons[0], 0.);
}

#[test]
fn calculate_ones() {
    let mut layer = Layer::new(3, 1, &CreationType::Constants { number: 2. });

    layer.calculate(Array1::ones([3]), &ActivationFunction::None);

    assert_eq!(layer.neurons[0], 8.);
}

#[test]
pub fn test_layer_mutation() {
    let mut layer = Layer::new(3, 1, &CreationType::Ones);

    layer.calculate(Array1::ones([3]), &ActivationFunction::None);
    
    let result_1 = layer.neurons[0];
    
    let mut rng = rand::thread_rng();
    layer.mutate(1., &mut rng);
    layer.calculate(Array1::ones([3]), &ActivationFunction::None);

    let result_2 = layer.neurons[0];

    assert_ne!(result_1, result_2);
}

#[test]
pub fn test_activation() {
    let mut layer = Layer::new(3, 1, &CreationType::Constants { number: 2. });

    layer.calculate(Array1::ones([3]), &ActivationFunction::ZeroOne);

    assert_eq!(layer.neurons[0], 1.);

    layer.calculate(Array1::ones([3]), &ActivationFunction::Positive);

    assert_eq!(layer.neurons[0], 8.);
}