use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone)]
struct Trade {
    symbol: String,
    position: i32,
    price: f64,
}

fn main() {
    // Sample trades
    let trades = vec![
        Trade { symbol: "AAPL".to_string(), position: 10, price: 150.0 },
        Trade { symbol: "GOOGL".to_string(), position: -5, price: 2800.0 },
        Trade { symbol: "MSFT".to_string(), position: 20, price: 330.0 },
        Trade { symbol: "TSLA".to_string(), position: -8, price: 700.0 },
    ];

    let trades = Arc::new(trades);
    let total_risk = Arc::new(Mutex::new(0.0));

    let mut handles = vec![];

    for trade in trades.iter().cloned() {
        let total_risk = Arc::clone(&total_risk);
        let handle = thread::spawn(move || {
            let risk = trade.position as f64 * trade.price;
            let mut total = total_risk.lock().unwrap();
            *total += risk;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Total Risk Exposure: {}", *total_risk.lock().unwrap());
}