pub trait Resource<'a> {
    type Global;

    fn new_global() -> Self::Global;
    fn new(g: &'a mut Self::Global) -> Self;
}

pub struct GlobalCounters {
    pub job_counter: usize,
}

pub struct Counters<'a> {
    pub job_counter: &'a usize,
}

impl<'a> Resource<'a> for Counters<'a> {
    type Global = GlobalCounters;

    fn new_global() -> Self::Global {
        GlobalCounters { job_counter: 0 }
    }

    fn new(g: &'a mut Self::Global) -> Self {
        Counters {
            job_counter: &g.job_counter,
        }
    }
}

pub fn my_anim(c: Counters) {
    println!("job {}", c.job_counter);
}

pub fn render<'a, R1, F>(anim: F)
where
    F: Fn(R1),
    R1: Resource<'a> + 'a,
{
    let mut g1 = R1::new_global();

    //.thread in 0..4 {
    //    let mut t1 = g1.new_thread(&mut r1);
    for _job in 0..4 {
        // I could not figure out how to do this without unsafe
        let g1m = unsafe { &mut *(&mut g1 as *mut R1::Global) };
        //let g1m = &mut g1;
        let f1 = R1::new(g1m);
        anim(f1);
    }

    //}
}

pub fn main5() {
    render(my_anim);
    //render::<GlobalCounters, _>(my_anim);
}
