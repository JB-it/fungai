use std::borrow::BorrowMut;

use ndarray::{Array1, arr2, arr1, s, Axis};
use crate::layer::*;
use rand::Rng;

#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub outputs: Array1<f32>,
    pub activation_function: ActivationFunction,
    pub fitness: f32,
    pub error: f32,
}

impl Network {
    pub fn new(sizes: Vec<usize>, creation_type: CreationType, activation: ActivationFunction) -> Self {
        let mut t_layers: Vec<Layer> = Vec::new();
        
        for i in 0..(sizes.len()-1) {
            let layer = Layer::new(sizes[i] as usize, sizes[i+1] as usize, &creation_type);
            t_layers.push(layer);
        }

        let t_output: Array1<f32> = Array1::<f32>::zeros([*sizes.last().unwrap()]);

        Network {
            layers: t_layers,
            outputs: t_output,
            activation_function: activation,
            fitness: 0.,
            error: 0.,
        }
    }

    pub fn calculate(&mut self, inputs: Vec<f32>) {
        let input_arr = Array1::from_vec(inputs);
        self.calculate_arr(input_arr);
    }

    pub fn calculate_arr(&mut self, inputs: Array1<f32>) {
        
        let mut result: Array1<f32> = inputs.clone();
        
        for mut layer in self.layers.iter_mut() {
            layer.calculate(result, &self.activation_function);
            result = layer.neurons.clone();
        }

        self.outputs = result;
    }

    pub fn reset(&mut self) {
        self.fitness = 0.;
        self.error = 0.;
    }

    pub fn mutate(&mut self, positions: i32, amount: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..positions {
            let layer_index: usize = rng.gen::<usize>() % self.layers.len();
            self.layers[layer_index].mutate(amount, &mut rng);
        }
    }

    pub fn print_network(&self) {
        for (i, l) in self.layers.iter().enumerate() {
            println!("Layer {}", i);
            l.print_layer();
        }
    }

    pub fn get_outputs(&self) -> Vec<f32> {
        self.outputs.to_vec().clone()
    }

}

#[test]
pub fn create_network() {
    let mut network = Network::new(vec![3, 1], CreationType::Zeroes, ActivationFunction::None);

    network.calculate_arr(Array1::zeros([3]));

    assert_eq!(network.outputs[0], 0.);
}

#[test]
pub fn test_single_layer() {
    let mut network = Network::new(vec![3, 1], CreationType::Ones, ActivationFunction::None);

    network.calculate_arr(Array1::ones([3]));

    assert_eq!(network.outputs[0], 4.);
}

#[test]
pub fn test_multiple_layers() {
    let mut network = Network::new(vec![3, 2, 1], CreationType::Ones, ActivationFunction::None);

    network.calculate_arr(Array1::ones([3]));

    assert_eq!(network.outputs[0], 9.);
}

#[test]
pub fn test_network_mutation() {
    let mut network = Network::new(vec![3, 2, 1], CreationType::Ones, ActivationFunction::None);
  
    network.calculate_arr(Array1::ones([3]));
    
    let result_1 = network.outputs[0];
    
    network.mutate(1, 1.);
    network.calculate_arr(Array1::ones([3]));

    let result_2 = network.outputs[0];

    assert_ne!(result_1, result_2);
}

#[test]
pub fn crude_learning_test() {
    let inputs = arr2(&[[1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.]
    ]);

    let outputs = arr1(&[1., 0., 0.]);
    
    let mut networks: Vec<Network> = Vec::new();
    let n_amount = 100;

    for _i in 0..n_amount {
        networks.push(Network::new(vec![3, 1], CreationType::Zeroes, ActivationFunction::Sigmoid));
    }


    let mut min_network = Network::new(vec![3, 1], CreationType::Zeroes, ActivationFunction::Sigmoid);

    
    for _i in 0..100 {

        for n in networks.iter_mut() {
            for i in 0..3 {
                n.calculate_arr(inputs.index_axis(Axis(0), i).to_owned());

                n.error += (outputs[i] - n.outputs[0]).abs();
            }
        }

        let mut min_error = 100.;
        let mut min_index: usize = 0;

        for (i, n) in networks.iter_mut().enumerate() {
            if n.error < min_error {
                min_error = n.error;
                min_index = i;
            }
        }

        min_network = networks[min_index].clone();
        min_network.reset();

        println!("{}", min_error);

        networks.clear();
        for _i in 0..n_amount {
            let mut n_network = min_network.clone();
            n_network.mutate(1, 1.);
            networks.push(n_network);
        }
        networks.push(min_network.clone());
    }

    let mut mn = min_network.borrow_mut();

    for i in 0..3 {
        mn.calculate_arr(inputs.index_axis(Axis(0), i).to_owned());
        println!("Input: {}\n Output: {}", outputs[i], mn.outputs[0]);
    }

    assert_eq!(min_network.error < 1., true);


    
}
