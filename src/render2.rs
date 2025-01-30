pub trait Animation {
    type RenderCtx;
    type FrameCtx;
    type Frame;

    fn make_render_ctx(&mut self) -> Self::RenderCtx;

    fn frame(&self, render_ctx: &mut Self::RenderCtx, frame_ctx: Self::FrameCtx) -> Option<Self::Frame>;
}

pub struct FrameDescriptor {
    pub frame: usize,
}

pub fn render(anim: &mut impl Animation<FrameCtx = FrameDescriptor, Frame = Vec<u8>>) {
    let mut render_ctx = anim.make_render_ctx();

    let mut i = 0;
    loop {
        let frame_ctx = FrameDescriptor { frame: i };
        let f = anim.frame(&mut render_ctx, frame_ctx);
        let Some(f) = f else {
            break;
        };
        println!("frame {}: {:?}", i, f);
        i += 1;
    }
}

pub struct TenZerosAnim {}

impl Animation for TenZerosAnim {
    type FrameCtx = FrameDescriptor;
    type RenderCtx = ();
    type Frame = Vec<u8>;

    fn make_render_ctx(&mut self) -> Self::RenderCtx {
        ()
    }

    fn frame(&self, _render_ctx: &mut Self::RenderCtx, _frame_ctx: FrameDescriptor) -> Option<Vec<u8>> {
        Some(vec![0; 10])
    }
}

pub struct ShortAnimation<A> {
    anim: A,
    len: usize,
}

impl<A: Animation<FrameCtx = FrameDescriptor>> Animation for ShortAnimation<A> {
    type FrameCtx = FrameDescriptor;
    type RenderCtx = A::RenderCtx;
    type Frame = A::Frame;

    fn make_render_ctx(&mut self) -> Self::RenderCtx {
        self.anim.make_render_ctx()
    }

    fn frame(&self, render_ctx: &mut Self::RenderCtx, frame_ctx: FrameDescriptor) -> Option<Self::Frame> {
        if frame_ctx.frame > self.len {
            return None;
        }
        self.anim.frame(render_ctx, frame_ctx)
    }
}

pub trait Shorten: Sized {
    fn shorten(self, len: usize) -> ShortAnimation<Self>;
}

impl<A: Animation<FrameCtx = FrameDescriptor>> Shorten for A {
    fn shorten(self, len: usize) -> ShortAnimation<A> {
        ShortAnimation { anim: self, len }
    }
}

pub fn main2() {
    let ten_zeros_anim = TenZerosAnim {};

    render(&mut ten_zeros_anim.shorten(10));
}
