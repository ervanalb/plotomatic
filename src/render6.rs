pub trait Resource {
    type GlobalResource;
    type FrameResource<'a>;

    fn new_global() -> Self::GlobalResource;
    fn new_frame<'a>(g: &'a mut Self::GlobalResource) -> Self::FrameResource<'a>;
}

pub struct GlobalCounters {
    pub job_counter: usize,
}

pub struct Counters<'a> {
    pub job_counter: &'a usize,
}

impl<'x> Resource for Counters<'x> {
    type GlobalResource = GlobalCounters;
    type FrameResource<'a> = Counters<'a>;

    fn new_global() -> Self::GlobalResource {
        GlobalCounters { job_counter: 0 }
    }

    fn new_frame<'a>(g: &'a mut Self::GlobalResource) -> Counters<'a> {
        Counters {
            job_counter: &g.job_counter,
        }
    }
}

pub fn my_anim(c: Counters) {
    println!("job {}", c.job_counter);
}

pub fn render<R1, F>(anim: F)
where
    F: for<'a> Fn(R1::FrameResource<'a>),
    R1: Resource,
{
    let mut g1 = R1::new_global();

    //.thread in 0..4 {
    //    let mut t1 = g1.new_thread(&mut r1);
    for _job in 0..4 {
        let f1 = R1::new_frame(&mut g1);
        anim(f1);
    }

    //}
}

pub fn main5() {
    //render<(my_anim);
    render::<Counters<'_>, _>(my_anim);
}
