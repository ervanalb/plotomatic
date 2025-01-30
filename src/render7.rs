use std::rc::Rc;

pub struct BasicFrameCtx {
    pub g: Rc<BasicGlobalCtx>,
    pub i: usize,
}

pub struct BasicGlobalCtx {
    pub fps: f64,
}

pub trait GlobalCtx: Sized {
    type FrameCtx: FrameCtx;

    fn new_frame_ctx(&self, i: usize) -> Self::FrameCtx;
}

pub trait Render<C: FrameCtx> {
    type Frame;
    type Output;

    fn render<A>(self, anim: A) -> Self::Output
    where
        A: Anim<C, Output = Self::Frame>;
}

pub trait FrameCtx: GlobalCtx<FrameCtx = Self> {
    fn i(&self) -> usize;
    fn fps(&self) -> f64;
    fn t(&self) -> f64 {
        self.i() as f64 / self.fps()
    }
}

impl GlobalCtx for BasicFrameCtx {
    type FrameCtx = BasicFrameCtx;

    fn new_frame_ctx(&self, i: usize) -> BasicFrameCtx {
        BasicFrameCtx {
            g: self.g.clone(),
            i,
        }
    }
}

impl FrameCtx for BasicFrameCtx {
    fn i(&self) -> usize {
        self.i
    }
    fn fps(&self) -> f64 {
        self.g.fps
    }
}

impl GlobalCtx for Rc<BasicGlobalCtx> {
    type FrameCtx = BasicFrameCtx;

    fn new_frame_ctx(&self, i: usize) -> BasicFrameCtx {
        BasicFrameCtx { g: self.clone(), i }
    }
}

fn render<A, C>(global_ctx: C, mut anim: A)
where
    A: Anim<<Rc<C> as GlobalCtx>::FrameCtx, Output = Vec<u8>>,
    Rc<C>: GlobalCtx,
{
    let global_ctx = Rc::new(global_ctx);
    for thread in 0..4 {
        for job in 0..20 {
            let i = thread * 20 + job;
            let frame_ctx = global_ctx.new_frame_ctx(i);
            if let Some(frame) = anim.render(frame_ctx) {
                println!("frame {:?}: {:?} (thread {:?})", i, frame, thread);
            }
        }
    }
}

pub trait Anim<C>
where
    C: FrameCtx,
{
    type Output;
    fn render(&mut self, frame: C) -> Option<Self::Output>;
}

impl<F, C, O> Anim<C> for F
where
    C: FrameCtx,
    F: FnMut(C) -> Option<O>,
{
    type Output = O;
    fn render(&mut self, ctx: C) -> Option<O> {
        self(ctx)
    }
}

pub fn shorten<A, C>(len: usize, mut anim: A) -> impl Anim<C, Output = A::Output>
where
    A: Anim<C>,
    C: FrameCtx,
{
    move |c: C| {
        if c.i() >= len {
            return None;
        }
        anim.render(c)
    }
}

pub fn concat<A0, A1, C>(mut anim0: A0, mut anim1: A1) -> impl Anim<C, Output = A0::Output>
where
    A0: Anim<C>,
    A1: Anim<C, Output = A0::Output>,
    C: FrameCtx,
{
    let mut clip: usize = 0;
    let mut start: usize = 0;
    move |c: C| {
        let i = c.i();
        loop {
            let c = c.new_frame_ctx(i - start);
            let output = match clip {
                0 => anim0.render(c),
                1 => anim1.render(c),
                _ => {
                    break None;
                }
            };
            if output.is_some() {
                break output;
            }
            start = i;
            clip += 1;
        }
    }
}

pub fn main7() {
    let a1 = shorten(20, |_: BasicFrameCtx| Some(vec![1, 2, 3]));
    let a2 = shorten(20, |_: BasicFrameCtx| Some(vec![4, 5, 6]));

    let a = concat(a1, a2);

    render(BasicGlobalCtx { fps: 60. }, a);
}
