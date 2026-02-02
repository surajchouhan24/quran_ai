// recorder-worklet.js
class RecorderWorklet extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(4096); // Buffer 4096 samples (~0.25s)
        this.offset = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0][0];
        if (!input) return true;

        for (let i = 0; i < input.length; i++) {
            this.buffer[this.offset++] = input[i];
            if (this.offset >= this.buffer.length) {
                // Send the chunk to the main thread
                this.port.postMessage(this.buffer);
                this.offset = 0;
            }
        }
        return true;
    }
}
registerProcessor('recorder-worklet', RecorderWorklet);