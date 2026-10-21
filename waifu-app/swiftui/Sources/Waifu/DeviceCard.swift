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

// The Device card, under the Model card: where the model runs, and -- for metal_disk_offload --
// how much of it stays on the GPU. Nothing is read from here: the next job reads the model where
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
                // The one choice that has to say what it costs: picked without being told, it is
                // a run that looks broken.
                .help(session.device == "metal_disk_offload"
                    ? "Leaves the weights in the model's files and reads each one off the disk as it is used: for a model larger than this Mac's memory, and slower."
                    : "Where the model runs")
            }
            .disabled(session.canStop)
            // Only for the one device that reads weights off the disk: every other keeps all of
            // them on the GPU already.
            if session.device == "metal_disk_offload" {
                SliderRow(
                    label: "vram (GB)", value: vramLimit, range: 1...macMemory, step: 0.5,
                    decimals: 1,
                    explanation: """
                        The most of the GPU's memory the model and its jobs may hold. The \
                        model's weights fill it as the model is read, the smallest first, and \
                        the rest are read off the disk as they are used. When a job needs more \
                        room than is left, the largest go back to the disk until it fits.

                        At the end of the slider there is no limit but this Mac's.

                        Takes effect when the model is next read.
                        """
                )
                .disabled(session.canStop)
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
