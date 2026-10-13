// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! The OpenMP runtime outliving the threads that use it.
//!
//! A libomp linked in statically -- the one the macOS build makes from LLVM's source -- shuts
//! itself down when the last thread registered with it exits, and starts again for the next one.
//! With threads coming and going at once, as the test harness's do, a shutdown runs into a start
//! and the process aborts in `___kmp_free` (`kmp_alloc.cpp`). In a file of its own, because the
//! case is a binary whose main thread has never asked OpenMP anything, which is what flint's
//! static initializer in `flint/operators.cc` is there to change.

use std::collections::VecDeque;
use std::thread;

use waifu::flint::{functional as F, Device, Tensor};

/// Short-lived threads, each running a parallel loop and exiting while others start: what the
/// harness does with a file of CPU tests, without the tests.
#[test]
fn threads_that_come_and_go_leave_openmp_running() {
    // Joined a few behind rather than each in turn: a join waits until the thread has exited, so
    // joining every thread before starting the next would keep its exit and the next one's start
    // from ever meeting.
    let mut running = VecDeque::new();
    for _ in 0..400 {
        running.push_back(thread::spawn(|| {
            let values: Vec<f32> = (0..64 * 256).map(|i| (i % 13) as f32 / 13.0).collect();
            let input = Tensor::from_f32(&[64, 256], &values).unwrap();
            assert_eq!(input.device(), Device::Cpu);

            let out = F::softmax(&input).unwrap();
            assert_eq!(out.shape(), vec![64, 256]);
        }));
        if running.len() > 4 {
            running.pop_front().unwrap().join().unwrap();
        }
    }
    for thread in running {
        thread.join().unwrap();
    }
}
