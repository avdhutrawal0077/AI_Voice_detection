/**
 * AudioChunkProcessor — AudioWorklet processor for TrueTone
 *
 * Buffers raw mono PCM audio, resamples to a guaranteed target sample rate
 * (default 16 000 Hz) using linear interpolation, and posts fixed-size
 * chunks to the main thread.
 *
 * Why resample here (in the worklet), not on the main thread?
 *   - The worklet already runs on the audio rendering thread and knows the
 *     actual native sampleRate global.
 *   - Resampling before postMessage means the main thread + VAD + WebSocket
 *     always see exactly targetSampleRate Hz PCM — no further conversion needed.
 *
 * Options (processorOptions):
 *   chunkDurationMs   {number}  Duration of each output chunk in ms. Default: 250.
 *   targetSampleRate  {number}  Target output sample rate in Hz. Default: 16000.
 *
 * Message format posted to main thread:
 *   {
 *     type:        'audio-chunk',
 *     samples:     Float32Array,   // resampled to targetSampleRate
 *     sampleCount: number,          // always chunkDurationMs * targetSampleRate / 1000
 *     sampleRate:  number,          // always targetSampleRate (e.g. 16000)
 *     captureTime: number           // audio-clock capture time (AudioContext.currentTime)
 *   }
 *
 * NOTE: This file must be served from a secure context (HTTPS or localhost).
 */

class AudioChunkProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        // --- Configuration ---
        const procOpts = (options && options.processorOptions) || {};
        this._chunkDurationMs   = procOpts.chunkDurationMs  || 250;
        this._targetSampleRate  = procOpts.targetSampleRate || 16000;

        // sampleRate is a global in the AudioWorkletGlobalScope — the actual
        // hardware/browser rate (e.g. 48000 on most desktop Chrome installs).
        this._nativeSampleRate = sampleRate;

        // Downsampling ratio: >1 means we need to downsample (e.g. 3.0 for 48k→16k).
        this._ratio = this._nativeSampleRate / this._targetSampleRate;

        // Native-rate buffer: accumulate until we have one full chunk of native samples.
        this._nativeChunkSize = Math.round(
            (this._chunkDurationMs / 1000) * this._nativeSampleRate
        );
        this._nativeBuffer = new Float32Array(this._nativeChunkSize);
        this._writeIndex = 0;

        // Output chunk size at the guaranteed target rate.
        // e.g. 250 ms * 16000 Hz / 1000 = 4000 samples — always.
        this._outputChunkSize = Math.round(
            (this._chunkDurationMs / 1000) * this._targetSampleRate
        );

        const resamplingNeeded = (this._nativeSampleRate !== this._targetSampleRate);

        console.info(
            `[AudioChunkProcessor] Initialized: ` +
            `native=${this._nativeSampleRate} Hz, ` +
            `target=${this._targetSampleRate} Hz, ` +
            `ratio=${this._ratio.toFixed(4)}, ` +
            `nativeChunk=${this._nativeChunkSize} samples, ` +
            `outputChunk=${this._outputChunkSize} samples, ` +
            `mode=${resamplingNeeded ? 'linear-interpolation-resample' : 'passthrough'}`
        );

        if (resamplingNeeded) {
            console.warn(
                `[AudioChunkProcessor] Resampling ${this._nativeSampleRate} Hz → ` +
                `${this._targetSampleRate} Hz (ratio ${this._ratio.toFixed(4)}). ` +
                `All posted chunks will carry sampleRate=${this._targetSampleRate}.`
            );
        }
    }

    /**
     * Linear-interpolation resampler.
     *
     * Converts a block of native-rate samples to the target rate.
     * For exact integer ratios (e.g. 48000→16000 = 3:1) this is lossless
     * for speech; for non-integer ratios it is a lightweight approximation
     * that is adequate for a speech-activity detector.
     *
     * @param {Float32Array} input — exactly this._nativeChunkSize samples
     * @returns {Float32Array}       — exactly this._outputChunkSize samples
     */
    _resample(input) {
        const output  = new Float32Array(this._outputChunkSize);
        const lastIdx = input.length - 1;

        for (let i = 0; i < this._outputChunkSize; i++) {
            // Map output-sample index back to a fractional position in the input.
            const pos  = i * this._ratio;
            const lo   = pos | 0;              // fast Math.floor for positive numbers
            const hi   = lo < lastIdx ? lo + 1 : lastIdx;
            const frac = pos - lo;
            output[i]  = input[lo] + frac * (input[hi] - input[lo]);
        }
        return output;
    }

    /**
     * Called by the Web Audio rendering thread ~every 128 frames.
     * inputs[0][0] is the first (mono) channel of the first input.
     */
    process(inputs /*, outputs, parameters */) {
        const input = inputs[0];
        if (!input || input.length === 0) {
            // No input connected — keep processor alive.
            return true;
        }

        // Use only the first channel (mono).
        const channelData = input[0];
        if (!channelData || channelData.length === 0) {
            return true;
        }

        let readIndex = 0;

        while (readIndex < channelData.length) {
            const remaining = this._nativeChunkSize - this._writeIndex;
            const available = channelData.length - readIndex;
            const toCopy    = Math.min(remaining, available);

            // Copy native-rate samples into the accumulation buffer.
            this._nativeBuffer.set(
                channelData.subarray(readIndex, readIndex + toCopy),
                this._writeIndex
            );

            this._writeIndex += toCopy;
            readIndex        += toCopy;

            // Flush when we have a complete native-rate chunk.
            if (this._writeIndex >= this._nativeChunkSize) {
                // Resample to target rate, or slice (passthrough) if already correct.
                const outputSamples = (this._nativeSampleRate !== this._targetSampleRate)
                    ? this._resample(this._nativeBuffer)
                    : this._nativeBuffer.slice();

                // captureTime is a global in AudioWorkletGlobalScope (audio-clock time).
                this.port.postMessage({
                    type:        'audio-chunk',
                    samples:     outputSamples,
                    sampleCount: this._outputChunkSize,
                    sampleRate:  this._targetSampleRate,  // always 16000 — guaranteed
                    captureTime: currentTime
                });

                this._writeIndex = 0;
            }
        }

        // Return true to keep the processor alive.
        return true;
    }
}

registerProcessor('audio-chunk-processor', AudioChunkProcessor);
