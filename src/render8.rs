use std::rc::Rc;

pub struct Frame {
    pub fps: Rc<f64>,
    pub i: usize,
    pub frame: Option<Vec<u8>>,
}

pub trait Counter {
    fn i(&self) -> usize;
    fn set_i(&mut self, i: usize);
    fn fps(&self) -> f64;
    fn t(&self) -> f64 {
        self.i() as f64 / self.fps()
    }
    fn has_frame(&self) -> bool;
    fn set_no_frame(&mut self);
}

impl Counter for Frame {
    fn i(&self) -> usize {
        self.i
    }
    fn set_i(&mut self, i: usize) {
        self.i = i;
    }
    fn fps(&self) -> f64 {
        *self.fps
    }
    fn has_frame(&self) -> bool {
        self.frame.is_some()
    }
    fn set_no_frame(&mut self) {
        self.frame = None;
    }
}

pub fn render<A>(fps: f64, mut anim: A)
where
    A: Anim<Frame>,
{
    let fps = Rc::new(fps);
    let mut i = 0;
    loop {
        let mut frame = Frame {
            fps: fps.clone(),
            i,
            frame: None,
        };
        anim.draw(&mut frame);
        let Some(frame) = frame.frame else {
            break;
        };
        println!("frame {:?}: {:?}", i, frame);
        i += 1;
    }
}

pub trait Anim<F>
where
    F: Counter,
{
    fn draw(&mut self, frame: &mut F);
}

impl<FN, FR> Anim<FR> for FN
where
    FR: Counter,
    FN: FnMut(&mut FR),
{
    fn draw(&mut self, frame: &mut FR) {
        self(frame);
    }
}

pub fn shorten<A, F>(len: usize, mut anim: A) -> impl Anim<F>
where
    A: Anim<F>,
    F: Counter,
{
    move |f: &mut F| {
        if f.i() >= len {
            f.set_no_frame();
            return;
        }
        anim.draw(f);
    }
}

pub fn concat<A0, A1, F>(mut anim0: A0, mut anim1: A1) -> impl Anim<F>
where
    A0: Anim<F>,
    A1: Anim<F>,
    F: Counter,
{
    let mut clip: usize = 0;
    let mut start: usize = 0;
    move |f: &mut F| {
        let i = f.i();
        loop {
            f.set_i(i - start);
            match clip {
                0 => {
                    anim0.draw(f);
                }
                1 => {
                    anim1.draw(f);
                }
                _ => {
                    break;
                }
            }
            if f.has_frame() {
                break;
            }
            start = i;
            clip += 1;
        }
    }
}

pub fn main7() {
    let a1 = shorten(20, |f: &mut Frame| {
        f.frame = Some(vec![1, 2, 3]);
    });
    let a2 = shorten(20, |f: &mut Frame| {
        f.frame = Some(vec![4, 5, 6]);
    });

    let a = concat(a1, a2);

    render(60., a);
}
