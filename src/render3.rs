use std::marker::PhantomData;

pub struct BasicAnimGlobalCtx {
    fps: f64,
}

pub struct BasicAnimCtx<'a> {
    global: &'a BasicAnimGlobalCtx,
    frame: usize,
}

pub trait BasicAnim {
    fn frame(&self) -> usize;
    fn t(&self) -> f64;
}

impl BasicAnim for BasicAnimCtx<'_> {
    fn frame(&self) -> usize {
        self.frame
    }
    fn t(&self) -> f64 {
        self.frame as f64 / self.global.fps
    }
}

pub struct ShortenedAnim<A, Context>
where
    A: Animation<Context>,
{
    anim: A,
    len: usize,
    _marker: PhantomData<Context>,
}

impl<A: Animation<Context>, Context: BasicAnim> Animation<Context> for ShortenedAnim<A, Context> {
    type Frame = A::Frame;

    fn frame(&self, ctx: Context) -> Option<Self::Frame> {
        if ctx.frame() > self.len {
            return None;
        }
        self.anim.frame(ctx)
    }
}

pub fn shorten<A: Animation<Context>, Context: BasicAnim>(
    anim: A,
    len: usize,
) -> ShortenedAnim<A, Context> {
    ShortenedAnim {
        anim,
        len,
        _marker: Default::default(),
    }
}

pub trait Animation<Context> {
    type Frame;

    fn frame<'a>(&self, ctx: Context) -> Option<Self::Frame>;
}

pub fn render<A>(global_ctx: BasicAnimGlobalCtx, anim: A)
where
    A: for<'a> Animation<BasicAnimCtx<'a>, Frame = Vec<u8>>,
{
    let mut i = 0;
    loop {
        let f = anim.frame(BasicAnimCtx {
            global: &global_ctx,
            frame: i,
        });
        let Some(f) = f else {
            break;
        };
        println!("frame {}: {:?}", i, f);
        i += 1;
    }
}

impl<R, F, C> Animation<C> for F
where
    F: Fn(C) -> Option<R>,
{
    type Frame = R;

    fn frame(&self, ctx: C) -> Option<Self::Frame> {
        (self)(ctx)
    }
}

pub fn main3() {
    struct ZeroAnim;

    impl Animation for ZeroAnim {
        type Frame = Vec<u8>;

        fn frame() -> Option<Self::Frame> {
            Some(vec![0; 10])
        }
    }

    let anim = ZeroAnim;
    //let anim = |_ctx: BasicAnimCtx<'_>| Some(vec![0; 10]);
    let a = shorten(anim, 2);

    render(BasicAnimGlobalCtx { fps: 60. }, a);
}
