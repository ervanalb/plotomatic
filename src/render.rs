use crate::output_writer::{Output, OutputTargetDescriptor, OutputWriter, OutputWriterError};
use rayon::prelude::*;
use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Condvar, Mutex};

pub struct Renderer {
    pub resolution: [usize; 2],
    pub fps: f64,
    pub output_target: OutputTargetDescriptor,
}

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub struct Id(NonZeroU64);

impl Default for Id {
    fn default() -> Self {
        Self(NonZeroU64::new(1).unwrap())
    }
}

impl<T> From<T> for Id
where
    T: std::hash::Hash,
{
    fn from(value: T) -> Self {
        Self(
            ahash::RandomState::with_seeds(1, 2, 3, 4)
                .hash_one(value)
                .try_into()
                .unwrap(),
        )
    }
}

impl Id {
    pub fn next(&self) -> Self {
        Self(self.0.checked_add(1).unwrap())
    }
}

#[derive(Default)]
pub struct Cache {
    map: BTreeMap<(Id, TypeId), Box<dyn Any + 'static>>,
}

impl Cache {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn reuse_or_create<V>(&mut self, id: impl Into<Id>, value_fn: impl Fn() -> V) -> Rc<V>
    where
        V: Any + 'static,
    {
        let id: Id = id.into();
        self.map
            .entry((id, TypeId::of::<V>()))
            .or_insert_with(|| Box::new(Rc::new(value_fn())))
            .downcast_ref::<Rc<V>>()
            .unwrap()
            .clone()
    }
}

impl Renderer {
    pub fn t(&self, i: usize) -> f64 {
        i as f64 / self.fps
    }

    pub fn from_time(&self, t: f64) -> usize {
        (t * self.fps) as usize
    }

    pub fn from_time_range(&self, range: Range<f64>) -> Range<usize> {
        let [start, end] = [range.start, range.end].map(|t| self.from_time(t));
        Range { start, end }
    }

    pub fn render<F>(
        &self,
        frame_range: Range<usize>,
        frame_function: F,
    ) -> Result<Output, OutputWriterError>
    where
        F: Sync + Send + Fn(usize, &mut Cache) -> Vec<u8>,
    {
        // Fire up the output writer
        let mut output_writer = OutputWriter::new(&self)?;

        // Channel with enough capacity to hold an item from each thread
        let (tx, rx) =
            mpsc::sync_channel(std::thread::available_parallelism().map_or(8, |n| n.get()));

        let gate = Arc::new((Mutex::new(0), Condvar::new()));

        rayon::scope(|s| {
            s.spawn(move |_| {
                frame_range
                    .par_bridge()
                    .map_init(|| Cache::new(), |c, i| (i, frame_function(i, c)))
                    .for_each_with((tx, gate), |(tx, gate), (i, frame_result)| {
                        let (lock, cond) = &**gate;

                        {
                            let mut guard =
                                cond.wait_while(lock.lock().unwrap(), |v| *v < i).unwrap();
                            let _ = tx.send(frame_result);
                            *guard = i + 1;
                        }

                        cond.notify_all();
                    });
            });

            // Write output
            for image_data in rx {
                output_writer.write(&image_data)?;
            }
            output_writer.close()
        })
    }
}
