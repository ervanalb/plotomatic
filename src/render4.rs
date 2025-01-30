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

pub struct ShortenedAnim<A>
where
    A: for<'a> Animation<Context<'a>: BasicAnim>,
{
    anim: A,
    len: usize,
}

impl<A> Animation for ShortenedAnim<A>
where
    for<'a> A: Animation<Context<'a>: BasicAnim> + 'a,
{
    type Frame = A::Frame;
    type Context<'a> = A::Context<'a>;

    fn frame(&self, ctx: Self::Context<'_>) -> Option<Self::Frame> {
        if ctx.frame() > self.len {
            return None;
        }
        self.anim.frame(ctx)
    }
}

pub fn shorten<A>(anim: A, len: usize) -> ShortenedAnim<A>
where
    for<'a> A: Animation<Context<'a>: BasicAnim> + 'a,
{
    ShortenedAnim { anim, len }
}

pub trait Animation {
    type Context<'a>;
    type Frame;

    fn frame<'a>(&self, ctx: Self::Context<'a>) -> Option<Self::Frame>;
}

pub fn render<A>(global_ctx: BasicAnimGlobalCtx, anim: A)
where
    A: for<'a> Animation<Context<'a> = BasicAnimCtx<'a>, Frame = Vec<u8>>,
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

pub fn main3() {
    struct ZeroAnim;

    impl Animation for ZeroAnim {
        type Context<'a> = BasicAnimCtx<'a>;
        type Frame = Vec<u8>;

        fn frame<'a>(&self, _ctx: Self::Context<'a>) -> Option<Self::Frame> {
            Some(vec![0; 10])
        }
    }

    let anim = ZeroAnim;
    let a = shorten(anim, 2);

    render(BasicAnimGlobalCtx { fps: 60. }, a);
}
