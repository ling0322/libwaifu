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

// The Device card, under the Model card: where the model runs, and -- on Metal -- the most of the
// GPU's memory it may hold. Nothing is read from here: the next job reads the model where
// this says, again if it has changed.

import SwiftUI

struct DeviceCard: View {
    @Bindable var session: Session

    var body: some View {
        FormGroup {
            LabeledContent("Device") {
                Picker("Device", selection: $session.device) {
                    ForEach(Session.devices, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden()
                .fillsRow()
                .help("Where the model runs")
            }
            .disabled(session.busy)
            // Only where the model runs on Metal, which auto is on a Mac: the processor has no GPU
            // memory to limit.
            if session.limitsVram {
                SliderRow(
                    label: "vram (GB)", value: vramLimit, range: 1...macMemory, step: 0.5,
                    decimals: 1)
                .disabled(session.busy)
            }
        } header: {
            Header("Device")
        }
    }

    /// The end of the slider, in gigabytes: this Mac's memory, in half gigabytes, rounded up so
    /// that the end is a step.
    private var macMemory: Double {
        max(1, (Double(ProcessInfo.processInfo.physicalMemory) / 1e9 * 2).rounded(.up) / 2)
    }

    /// The limit as the slider has it: no limit is the end of it, and the end of it is no limit.
    private var vramLimit: Binding<Double> {
        Binding(
            get: { session.vramLimitGB == 0 ? macMemory : min(session.vramLimitGB, macMemory) },
            set: { session.vramLimitGB = $0 >= macMemory ? 0 : $0 })
    }
}
