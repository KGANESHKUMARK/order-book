use std::collections::{BTreeMap, HashMap};
use ordered_float::OrderedFloat;

#[derive(Debug, Clone)]
struct Order {
    id: u64,
    price: f64,
    _quantity: u32, // Prefix with underscore to suppress warning
}

#[derive(Debug)]
struct OrderBook {
    buy_orders: BTreeMap<OrderedFloat<f64>, Vec<Order>>,  // Orders sorted by price descending
    sell_orders: BTreeMap<OrderedFloat<f64>, Vec<Order>>, // Orders sorted by price ascending
    orders_by_id: HashMap<u64, Order>,      // Fast lookup by order ID
}

impl OrderBook {
    fn new() -> Self {
        Self {
            buy_orders: BTreeMap::new(),
            
            sell_orders: BTreeMap::new(),
            orders_by_id: HashMap::new(),
        }
    }

    fn add_order(&mut self, id: u64, price: f64, _quantity: u32, is_buy: bool) {
        let order = Order { id, price, _quantity };
        let book = if is_buy {
            &mut self.buy_orders
        } else {
            &mut self.sell_orders
        };
        book.entry(OrderedFloat(price)).or_default().push(order.clone());
        self.orders_by_id.insert(id, order);
    }

    fn cancel_order(&mut self, id: u64) {
        if let Some(order) = self.orders_by_id.remove(&id) {
            let book = if self.buy_orders.contains_key(&OrderedFloat(order.price)) {
                &mut self.buy_orders
            } else {
                &mut self.sell_orders
            };
            if let Some(orders) = book.get_mut(&OrderedFloat(order.price)) {
                orders.retain(|o| o.id != id);
            }
        }
    }
}

fn main() {
    let mut book = OrderBook::new();
    book.add_order(1, 100.0, 10, true); // Add a buy order
    book.add_order(2, 101.0, 5, false); // Add a sell order
    println!("{:?}", book);
    book.cancel_order(1); // Cancel order
    println!("{:?}", book);
}