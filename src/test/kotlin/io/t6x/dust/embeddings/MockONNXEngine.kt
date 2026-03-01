package io.t6x.dust.embeddings

import io.t6x.dust.onnx.ONNXEngine
import io.t6x.dust.onnx.ONNXTensorMetadata
import io.t6x.dust.onnx.TensorData

class MockONNXEngine(
    override val inputMetadata: List<ONNXTensorMetadata> = listOf(
        ONNXTensorMetadata("input_ids", listOf(-1, -1), "int64"),
        ONNXTensorMetadata("attention_mask", listOf(-1, -1), "int64"),
        ONNXTensorMetadata("token_type_ids", listOf(-1, -1), "int64"),
    ),
    override val outputMetadata: List<ONNXTensorMetadata> = listOf(
        ONNXTensorMetadata("last_hidden_state", listOf(-1, -1, 2), "float32"),
    ),
    override val accelerator: String = "cpu",
    var cannedOutput: Map<String, TensorData>? = null,
    var outputGenerator: ((Map<String, TensorData>) -> Map<String, TensorData>)? = null,
    var errorToThrow: Exception? = null,
) : ONNXEngine {
    var lastInputs: Map<String, TensorData>? = null
        private set
    var runCallCount = 0
        private set
    var closeCallCount = 0
        private set

    override fun run(inputs: Map<String, TensorData>): Map<String, TensorData> {
        lastInputs = inputs
        runCallCount += 1
        errorToThrow?.let { throw it }
        outputGenerator?.let { return it(inputs) }
        cannedOutput?.let { return it }
        return dynamicOutput(inputs)
    }

    override fun close() {
        closeCallCount += 1
    }

    private fun dynamicOutput(inputs: Map<String, TensorData>): Map<String, TensorData> {
        val inputIds = inputs["input_ids"] ?: inputs.values.firstOrNull()
        val batchSize = inputIds?.shape?.getOrNull(0) ?: 1
        val seqLen = inputIds?.shape?.getOrNull(1) ?: 1
        val outputName = outputMetadata.firstOrNull()?.name ?: "last_hidden_state"
        val data = mutableListOf<Double>()

        repeat(batchSize) {
            for (tokenIndex in 0 until seqLen) {
                if (tokenIndex == seqLen - 1) {
                    data += 100.0
                    data += 100.0
                } else {
                    data += (tokenIndex + 1).toDouble()
                    data += 0.0
                }
            }
        }

        return mapOf(
            outputName to TensorData(
                name = outputName,
                dtype = "float32",
                shape = listOf(batchSize, seqLen, 2),
                data = data,
            ),
        )
    }
}
