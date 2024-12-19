use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{array, Array1};

fn main() {
    // Dataset
    let x = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.8], [1.0, 0.6]];
    let y: Array1<_> = array![0, 0, 1, 1].into();
    let dataset = Dataset::from((x, y));

    // Model training
    let model = LogisticRegression::default().fit(&dataset).unwrap();

    // Prediction
    let pred = model.predict(&dataset.records());
    println!("Prediction: {:?}", pred);
}