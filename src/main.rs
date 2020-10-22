use decimal::d128;
use std::time::Instant;

// test problem: y' = 2xy
fn yprime(x: d128, y: d128) -> d128 {
    d128!(2) * x * y
}

// exact solution: y = y0 * e^(x^2)
fn yexact(x: d128, y0: d128) -> d128 {
    y0 * (x * x).exp()
}

// we love our RK4, don't we folks
fn runge_kutta(x0: d128, y0: d128, h: d128, n: usize) -> Vec<(d128, d128)> {
    let mut approx: Vec<(d128, d128)> = Vec::new();
    approx.push((x0, y0));
    for _ in 1..=n {
        // k1 = y'(xi, yi)
        let k1 = yprime(approx.last().unwrap().0, approx.last().unwrap().1);
        // k2 = y'(xi + h/2, yi + (hk1)/2)
        let k2 = yprime(
            approx.last().unwrap().0 + h / d128!(2),
            approx.last().unwrap().1 + (h * k1) / d128!(2),
        );
        // k3 = y'(xi + h/2, yi + (hk1)/2)
        let k3 = yprime(
            approx.last().unwrap().0 + h / d128!(2),
            approx.last().unwrap().1 + (h * k2) / d128!(2),
        );
        // k4 = y'(xi + h, yi + hk3)
        let k4 = yprime(
            approx.last().unwrap().0 + h,
            approx.last().unwrap().1 + (h * k3),
        );
        // y(i+1) = yi + (h/6)(k1 + 2*k2 + 2*k3 + k4)
        approx.push((
            approx.last().unwrap().0 + h,
            approx.last().unwrap().1 + h * (k1 + d128!(2) * k2 + d128!(2) * k3 + k4) / d128!(6),
        ));
    }
    approx
}

// adams-bashforth 4-step method
fn adams(x0: d128, y0: d128, h: d128, n: usize) -> Vec<(d128, d128)> {
    // adams method is multistep; needs to be initialized by another method, like rk4
    let mut approx: Vec<(d128, d128)> = runge_kutta(x0, y0, h, 3);
    for _ in 4..=n {
        // y(i+1) = yi + (h/24)(55y'(xi,yi) - 59y'(x(i-1), y(i-1)) + 37y'(x(i-2), y(i-2)) - 9y'(x(i-3), y(i-3)))
        approx.push((
            approx.last().unwrap().0 + h,
            approx.last().unwrap().1
                + h * (d128!(55)
                    * yprime(
                        approx.get(approx.len() - 1).unwrap().0,
                        approx.get(approx.len() - 1).unwrap().1,
                    )
                    - d128!(59)
                        * yprime(
                            approx.get(approx.len() - 2).unwrap().0,
                            approx.get(approx.len() - 2).unwrap().1,
                        )
                    + d128!(37)
                        * yprime(
                            approx.get(approx.len() - 3).unwrap().0,
                            approx.get(approx.len() - 3).unwrap().1,
                        )
                    - d128!(9)
                        * yprime(
                            approx.get(approx.len() - 4).unwrap().0,
                            approx.get(approx.len() - 4).unwrap().1,
                        ))
                    / d128!(24),
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
    // using y(0) = 6 for our initial condition
    let x0 = d128!(0);
    let y0 = d128!(6);
    // h = 0.01
    let h = d128!(1) / d128!(1000);
    let n = 2000;

    let now = Instant::now();
    let runge = runge_kutta(x0, y0, h, n);
    let elapsed = now.elapsed();

    let now_a = Instant::now();
    let adams = adams(x0, y0, h, n);
    let elapsed_a = now_a.elapsed();

    let now_e = Instant::now();
    let exact = yexact_vec(x0, y0, h, n);
    let elapsed_e = now_e.elapsed();

    println!("Exact solution time elapsed: {} ms", elapsed_e.as_millis());
    println!(
        "RK err at {}: {}",
        runge.last().unwrap().0,
        runge.last().unwrap().1 - exact.last().unwrap().1
    );
    println!("Time elapsed: {} ms", elapsed.as_millis());

    println!(
        "Adams err at {}: {}",
        adams.last().unwrap().0,
        adams.last().unwrap().1 - exact.last().unwrap().1
    );
    println!("Time elapsed: {} ms", elapsed_a.as_millis());
}
