mod layer;

use ndarray::{Array1, Array};
use layer::*;
use rand::Rng;

struct Network {
    pub layers: Vec<Layer>,
    pub outputs: Array1<f32>,
    pub fitness: f32,
}

impl Network {
    pub fn new(sizes: Vec<usize>, creation_type: CreationType) -> Self {
        let mut t_layers: Vec<Layer> = Vec::new();
        
        for i in 0..(sizes.len()-1) {
            let layer = Layer::new(sizes[i] as usize, sizes[i+1] as usize, &creation_type);
            t_layers.push(layer);
        }

        let t_output: Array1<f32> = Array1::<f32>::zeros([*sizes.last().unwrap()]);

        Network {
            layers: t_layers,
            outputs: t_output,
            fitness: 0.,
        }
    }

    pub fn calculate(&mut self, inputs: Array1<f32>) {
        let mut result: Array1<f32> = inputs.clone();
        
        for mut layer in self.layers.iter_mut() {
            layer.calculate(result);
            result = layer.neurons.clone();
        }

        self.outputs = result;
    }

    pub fn mutate(&mut self, positions: i32, amount: f32) {
        let mut rng = rand::thread_rng();
        for i in 0..positions {
            let layer_index: usize = rng.gen::<usize>() % self.layers.len();
            self.layers[layer_index].mutate(amount, &mut rng);
        }
    }

}

#[test]
pub fn create_network() {
    let mut network = Network::new(vec![3, 1], CreationType::Zeroes);

    network.calculate(Array1::zeros([3]));

    assert_eq!(network.outputs[0], 0.);
}

#[test]
pub fn test_single_layer() {
    let mut network = Network::new(vec![3, 1], CreationType::Ones);

    network.calculate(Array1::ones([3]));

    assert_eq!(network.outputs[0], 4.);
}

#[test]
pub fn test_multiple_layers() {
    let mut network = Network::new(vec![3, 2, 1], CreationType::Ones);

    network.calculate(Array1::ones([3]));

    assert_eq!(network.outputs[0], 9.);
}

#[test]
pub fn test_network_mutation() {
    let mut network = Network::new(vec![3, 2, 1], CreationType::Ones);
  
    network.calculate(Array1::ones([3]));
    
    let result_1 = network.outputs[0];
    
    network.mutate(1, 1.);
    network.calculate(Array1::ones([3]));

    let result_2 = network.outputs[0];

    assert_ne!(result_1, result_2);
}