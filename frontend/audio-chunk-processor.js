/**
 * AudioChunkProcessor — AudioWorklet processor for TrueTone
 *
 * Buffers raw mono PCM audio and posts fixed-size chunks to the main thread.
 * Chunk duration is configurable via processorOptions.chunkDurationMs.
 *
 * Message format posted to main thread:
 *   { type: 'audio-chunk', samples: Float32Array, sampleCount: number, sampleRate: number }
 *
 * NOTE: This file must be served from a secure context (HTTPS or localhost).
 */

class AudioChunkProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        // --- Configuration ---
        const procOpts = (options && options.processorOptions) || {};
        this._chunkDurationMs = procOpts.chunkDurationMs || 250;

        // sampleRate is a global in the AudioWorkletGlobalScope
        this._sampleRate = sampleRate;
        this._chunkSize = Math.round((this._chunkDurationMs / 1000) * this._sampleRate);

        // --- Internal buffer ---
        this._buffer = new Float32Array(this._chunkSize);
        this._writeIndex = 0;

        // Log config once on creation
        // (console.log inside worklet goes to browser console)
        console.info(
            `[AudioChunkProcessor] Initialized: ` +
            `chunkDuration=${this._chunkDurationMs}ms, ` +
            `sampleRate=${this._sampleRate}, ` +
            `chunkSize=${this._chunkSize} samples`
        );
    }

    /**
     * Called by the Web Audio rendering thread ~every 128 frames.
     * inputs[0][0] is the first (mono) channel of the first input.
     */
    process(inputs /*, outputs, parameters */) {
        const input = inputs[0];
        if (!input || input.length === 0) {
            // No input connected — keep processor alive
            return true;
        }

        // Use only the first channel (mono)
        const channelData = input[0];
        if (!channelData || channelData.length === 0) {
            return true;
        }

        let readIndex = 0;

        while (readIndex < channelData.length) {
            const remaining = this._chunkSize - this._writeIndex;
            const available = channelData.length - readIndex;
            const toCopy = Math.min(remaining, available);

            // Copy samples into the internal buffer
            this._buffer.set(
                channelData.subarray(readIndex, readIndex + toCopy),
                this._writeIndex
            );

            this._writeIndex += toCopy;
            readIndex += toCopy;

            // Flush when buffer is full
            if (this._writeIndex >= this._chunkSize) {
                // Transfer a copy so the buffer can be reused immediately
                // currentTime is a global in AudioWorkletGlobalScope — audio-clock capture time
                this.port.postMessage({
                    type: 'audio-chunk',
                    samples: this._buffer.slice(),
                    sampleCount: this._chunkSize,
                    sampleRate: this._sampleRate,
                    captureTime: currentTime
                });

                this._writeIndex = 0;
            }
        }

        // Return true to keep the processor alive
        return true;
    }
}

registerProcessor('audio-chunk-processor', AudioChunkProcessor);
