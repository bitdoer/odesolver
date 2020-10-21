use decimal::d128;
use std::time::Instant;

// test problem: y' = 2y
// (x is included in parameters so that i don't have to refactor the code for, say,
// a problem like y' = 2xy)
fn yprime(x: d128, y: d128) -> d128 {
    d128::from(2) * y
}

// exact solution: y = y0*e^(2x)
fn yexact(x: d128, y0: d128) -> d128 {
    y0 * (d128::from(2) * x).exp()
}

// we love our RK4, don't we folks
fn runge_kutta(x0: d128, y0: d128, h: d128, n: usize) -> Vec<(d128, d128)> {
    let mut approx: Vec<(d128, d128)> = Vec::new();
    approx.push((x0, y0));
    for _ in 1..=n {
        // k1 = y'(xi, yi)
        let k1 = yprime(
            approx.last().unwrap().0,
            approx.last().unwrap().1,
        );
        // k2 = y'(xi + h/2, yi + (hk1)/2)
        let k2 = yprime(
            approx.last().unwrap().0 + h / d128::from(2),
            approx.last().unwrap().1 + (h * k1) / d128::from(2),
        );
        // k3 = y'(xi + h/2, yi + (hk1)/2)
        let k3 = yprime(
            approx.last().unwrap().0 + h / d128::from(2),
            approx.last().unwrap().1 + (h * k2) / d128::from(2),
        );
        // k4 = y'(xi + h, yi + hk3)
        let k4 = yprime(
            approx.last().unwrap().0 + h,
            approx.last().unwrap().1 + (h * k3),
        );
        // y(i+1) = yi + (h/6)(k1 + 2*k2 + 2*k3 + k4)
        approx.push((
            approx.last().unwrap().0 + h,
            approx.last().unwrap().1 + h * (k1 + d128::from(2) * k2 + d128::from(2) * k3 + k4) / d128::from(6),
        ));
    }
    approx
}

fn yexact_vec(x0: d128, y0: d128, h: d128, n: usize) -> Vec<(d128, d128)> {
    let mut exact: Vec<(d128, d128)> = Vec::new();
    exact.push((x0, y0));
    for _ in 1..=n {
        let x = exact.last().unwrap().0 + h;
        exact.push((x, yexact(x, y0)));
    }
    exact
}

fn main() {
    // using y(0) = 1 for our initial condition
    let x0 = d128::from(0);
    let y0 = d128::from(1);
    // h = 0.01
    let h = d128::from(1) / d128::from(100);
    let n = 200;

    let now = Instant::now();

    let approx = runge_kutta(x0, y0, h, n);
    let exact = yexact_vec(x0, y0, h, n);

    let elapsed = now.elapsed();

    for i in 0..=n {
        println!(
            "err at {}: {} --- exact: {}",
            approx.get(i).unwrap().0,
            (approx.get(i).unwrap().1 - exact.get(i).unwrap().1),
            exact.get(i).unwrap().1
        );
    }

    println!("Time elapsed: {} ns", elapsed.as_nanos());
}