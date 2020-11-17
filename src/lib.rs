use rand::distributions::{Distribution, Standard};
use std::{fmt::Debug, marker::PhantomData};

/// A StaticGeneratorOf<T> can generate a T but cannot be parametrized
pub trait StaticGeneratorOf<T> {
    fn generate<R: rand::Rng + ?Sized>(rng: &mut R) -> T;
}

/// A GeneratorOf<T> can randomly produce a value of type T.
/// The `generate` method uses the instance, so it can be parametrized
/// with bounds, for example.
pub trait GeneratorOf<T> {
    fn generate<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> T;
}

impl<T, G> GeneratorOf<T> for G
where
    G: StaticGeneratorOf<T>,
{
    fn generate<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> T {
        <G as StaticGeneratorOf<T>>::generate(rng)
    }
}

impl<T> StaticGeneratorOf<T> for T
where
    Standard: Distribution<T>,
{
    fn generate<R: rand::Rng + ?Sized>(rng: &mut R) -> T {
        rng.gen()
    }
}

pub struct VecBounded<T, G>(usize, usize, PhantomData<T>, PhantomData<G>);

impl<T, G: StaticGeneratorOf<T>> VecBounded<T, G> {
    pub fn new(low: usize, high: usize) -> VecBounded<T, G> {
        VecBounded(low, high, PhantomData, PhantomData)
    }
}

impl<T, G> GeneratorOf<Vec<T>> for VecBounded<T, G>
where
    G: StaticGeneratorOf<T>,
{
    fn generate<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec<T> {
        let VecBounded(low, high, _, _) = self;
        let length: usize = rng.gen_range(low, high);

        let mut result = Vec::new();
        result.reserve(length);
        for _ in 0..length {
            result.push(G::generate(rng));
        }
        result
    }
}

#[allow(dead_code)]
struct ResultGen<T, E>(PhantomData<T>, PhantomData<E>);

impl<T, E> StaticGeneratorOf<Result<T, E>> for ResultGen<T, E>
where
    Standard: Distribution<T> + Distribution<E>,
{
    fn generate<R: rand::Rng + ?Sized>(rng: &mut R) -> Result<T, E> {
        if rng.gen_bool(0.5) {
            Ok(rng.gen())
        } else {
            Err(rng.gen())
        }
    }
}

///

type Pred<'a, T> = Box<dyn Fn(&T) -> bool + 'a>;

pub struct When<'a, T: 'a>(Pred<'a, T>);

impl<'a, T> When<'a, T> {
    pub fn new(f: Pred<T>) -> When<T> {
        When(f)
    }

    pub fn test(&self, t: &T) -> bool {
        self.0(t)
    }

    pub fn or<F: Fn(&T) -> bool + 'a>(self, f: F) -> When<'a, T> {
        When(Box::new(move |t| self.0(t) || f(t)))
    }

    pub fn and<F: Fn(&T) -> bool + 'a>(self, f: F) -> When<'a, T> {
        When(Box::new(move |t| self.0(t) && f(t)))
    }
}

pub fn when<'a, T, F: Fn(&T) -> bool + 'a>(f: F) -> When<'a, T> {
    When::new(Box::new(f))
}

pub fn always<'a, T>() -> When<'a, T> {
    when(|_| true)
}

///

pub struct Test<'a, T, F> {
    pub count: u32,
    pub when: When<'a, T>,
    pub is_passing: F,
}

impl<'a, T: Debug, F: Fn(&T) -> bool + 'a> Test<'a, T, F> {
    pub fn new(count: u32, when: When<'a, T>, is_passing: F) -> Test<'a, T, F> {
        return Test {
            count,
            when,
            is_passing,
        };
    }

    pub fn run_static<G: StaticGeneratorOf<T>>(self, msg: &'static str) {
        let Test {
            count,
            when,
            is_passing,
        } = self;

        let mut rng = rand::thread_rng();
        for _ in 0..count {
            let t = G::generate(&mut rng);
            if when.test(&t) {
                assert!(
                    is_passing(&t),
                    "Property not satisfied for {:?}: {}",
                    t,
                    msg
                );
            }
        }
    }

    pub fn run<G: GeneratorOf<T>>(self, generator: G, msg: &str) {
        let Test {
            count,
            when,
            is_passing,
        } = self;
        let mut rng = rand::thread_rng();
        for _ in 0..count {
            let t = generator.generate(&mut rng);
            if when.test(&t) {
                assert!(
                    is_passing(&t),
                    "Property not satisfied for {:?}: {}",
                    t,
                    msg
                );
            }
        }
    }
}

///

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let test = Test::new(1000, when(|x| *x > 0), |x| 0 < *x);
        test.run_static::<i8>("(x > 0) => (0 < x)");
    }

    #[test]
    fn sorting_works() {
        let test = Test::new(1000, always(), |xs: &Vec<u8>| {
            let mut ys = xs.clone();
            ys.sort();
            for i in 0..ys.len().max(1) - 1 {
                if ys[i] > ys[i + 1] {
                    return false;
                }
            }
            true
        });
        test.run(
            VecBounded::<u8, u8>::new(0, 20),
            "After sorting a Vec, it's sorted",
        );
    }

    #[test]
    fn associativity_works() {
        let are_finite =
            when(|t: &(f64, f64, f64)| t.0.is_finite())
            .and(|t| t.1.is_finite())
            .and(|t| t.2.is_finite());
        let test = Test::new(
            100000,
             are_finite,
              |(x, y, z)| {
            let diff = ((x + y) + z) - (x + (y + z));
            diff < 0.000000000001
        });
        test.run_static::<(f64, f64, f64)>("(x + y) + z == x + (y + z)")
    }
}
