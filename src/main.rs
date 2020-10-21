use bigdecimal::BigDecimal;
use num_bigint::BigInt;

type BD = BigDecimal;

// test problem: y' = 2y
// (x is included in parameters so that i don't have to refactor the code for, say,
// a problem like y' = 2xy)
fn yprime(x: BD, y: BD) -> BD {
    BD::from(2) * y
}

// exact solution: y = y0*e^(2x)
fn yexact(x: BD, y0: BD) -> BD {
    y0 * (BD::from(2) * x).exp()
}

// we love our RK4, don't we folks
fn runge_kutta(x0: BD, y0: BD, h: BD, n: usize) -> Vec<(BD, BD)> {
    let mut approx: Vec<(BD, BD)> = Vec::new();
    // BigDecimal doesn't implement Copy for obvious reasons, so we have a lot of cloning to do;
    // this program has dogshit optimization for that reason
    approx.push((x0.clone(), y0.clone()));
    for _ in 1..=n {
        // k1 = y'(xi, yi)
        let k1 = yprime(
            approx.clone().last().unwrap().0.clone(),
            approx.clone().last().unwrap().1.clone(),
        );
        // k2 = y'(xi + h/2, yi + (hk1)/2)
        let k2 = yprime(
            approx.clone().last().unwrap().0.clone() + h.clone() / BD::from(2),
            approx.clone().last().unwrap().1.clone() + h.clone() * k1.clone() / BD::from(2),
        );
        // k3 = y'(xi + h/2, yi + (hk1)/2)
        let k3 = yprime(
            approx.clone().last().unwrap().0.clone() + h.clone() / BD::from(2),
            approx.clone().last().unwrap().1.clone() + h.clone() * k2.clone() / BD::from(2),
        );
        // k4 = y'(xi + h, yi + hk3)
        let k4 = yprime(
            approx.clone().last().unwrap().0.clone() + h.clone(),
            approx.clone().last().unwrap().1.clone() + h.clone() * k3.clone(),
        );
        // y(i+1) = yi + (h/6)(k1 + 2*k2 + 2*k3 + k4)
        approx.push((
            approx.clone().last().unwrap().0.clone() + h.clone(),
            approx.clone().last().unwrap().1.clone()
                + h.clone()
                    * (k1.clone()
                        + BD::from(2) * k2.clone()
                        + BD::from(2) * k3.clone()
                        + k4.clone())
                    / BD::from(6),
        ));
    }
    approx
}

fn yexact_vec(x0: BD, y0: BD, h: BD, n: usize) -> Vec<(BD, BD)> {
    let mut exact: Vec<(BD, BD)> = Vec::new();
    exact.push((x0.clone(), y0.clone()));
    for _ in 1..=n {
        let x = exact.clone().last().unwrap().0.clone() + h.clone();
        exact.push((x.clone(), yexact(x.clone(), y0.clone())));
    }
    exact
}

fn main() {
    // using y(0) = 1 for our initial condition
    let x0 = BD::from(0);
    let y0 = BD::from(1);
    // h = 0.05
    let h = BD::new(BigInt::from(5), 2);
    let n = 40;
    let approx = runge_kutta(x0.clone(), y0.clone(), h.clone(), n);
    let exact = yexact_vec(x0.clone(), y0.clone(), h.clone(), n);

    for i in 0..=n {
        println!(
            "err at {}: {}",
            approx.clone().get(i).unwrap().0.clone().with_prec(3),
            (approx.clone().get(i).unwrap().1.clone() - exact.clone().get(i).unwrap().1.clone())
                .with_prec(10)
        );
    }
}