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

// What the bar says a job is doing. libwaifu says which stage it is on and how far into it, and
// the words are this app's; a download comes with the model manager's own.

import WaifuKit

extension JobProgress {
    /// The status line: "step 3 of 20", "decoding the layers, 4 of 13". `model` is what a read of
    /// weights says it is reading.
    func line(reading model: String? = nil) -> String {
        // Said before each one, with `done` already done: the one under way is the next.
        let at = "\(min(done + 1, total)) of \(total)"
        let counted = total > 0
        switch stage {
        case .fetching: return words ?? "fetching"
        case .loadingWeights: return model.map { "reading \($0)" } ?? "reading the model"

        case .imageGenerationEncodingPrompt: return "reading the prompt"
        case .imageGenerationDenoising: return counted ? "step \(done) of \(total)" : "drawing"
        case .imageGenerationDecodingImage: return "making the picture"

        // "of about": how many tokens there will be is the model's to find out, and the number is
        // worked out from the length of the text. A bar that said "of" and then went past it
        // would be lying.
        case .textToSpeechReadingText: return "reading the text"
        case .textToSpeechGeneratingTokens: return counted ? "token \(done) of about \(total)" : "saying it"
        case .textToSpeechSynthesizingAudio: return "making the sound"

        case .voiceConversionReadingRecordings: return "listening to the recordings"
        case .voiceConversionGeneratingTokens: return counted ? "token \(done) of about \(total)" : "saying it"
        case .voiceConversionDenoising: return counted ? "step \(done) of \(total)" : "drawing the voice"
        case .voiceConversionSynthesizingAudio: return total > 1 ? "making the sound, \(at)" : "making the sound"

        case .layerDecompositionEncodingImage: return "encoding the picture"
        case .layerDecompositionDenoisingLayers:
            return counted ? "drawing the layers, step \(done) of \(total)" : "drawing the layers"
        case .layerDecompositionDecodingLayers: return counted ? "decoding the layers, \(at)" : "decoding the layers"
        case .layerDecompositionEncodingHead: return "encoding the head"
        case .layerDecompositionDenoisingHeadLayers:
            return counted ? "drawing the head's layers, step \(done) of \(total)" : "drawing the head's layers"
        case .layerDecompositionDecodingHeadLayers:
            return counted ? "decoding the head's layers, \(at)" : "decoding the head's layers"
        case .layerDecompositionEncodingForDepth:
            return counted ? "encoding the layers for depth, \(at)" : "encoding the layers for depth"
        case .layerDecompositionDenoisingDepth:
            return counted ? "ordering the layers, step \(done) of \(total)" : "ordering the layers"
        case .layerDecompositionDecodingDepth: return counted ? "decoding the depth, \(at)" : "decoding the depth"
        case .layerDecompositionCuttingIntoParts: return "cutting the layers into parts"

        // Nothing here asks for these yet, or a newer library says something this does not know.
        default: return "working"
        }
    }
}
