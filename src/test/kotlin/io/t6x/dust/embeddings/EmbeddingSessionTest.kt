package io.t6x.dust.embeddings

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.SessionPriority
import io.t6x.dust.onnx.ONNXError
import io.t6x.dust.onnx.ONNXModelMetadataValue
import io.t6x.dust.onnx.ONNXSession
import io.t6x.dust.onnx.ONNXTensorMetadata
import io.t6x.dust.onnx.TensorData
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

class EmbeddingSessionTest {
    @Test
    fun e2T6EmbedRunsTokenizeInferPoolAndNormalize() {
        val engine = MockONNXEngine()
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = ONNXSession(
                sessionId = "mini-embed",
                engine = engine,
                metadata = ONNXModelMetadataValue(
                    inputs = engine.inputMetadata,
                    outputs = engine.outputMetadata,
                    accelerator = "cpu",
                    opset = 17,
                ),
                sessionPriority = SessionPriority.INTERACTIVE,
            ),
            config = makeConfig(),
        )

        val result = session.embed("hello")

        assertEquals("mini-embed", result.modelId)
        assertEquals(3, result.tokenCount)
        assertFalse(result.truncated)
        assertEquals(1f, result.embedding[0], 0.0001f)
        assertEquals(0f, result.embedding[1], 0.0001f)
        assertEquals("int64", engine.lastInputs?.get("input_ids")?.dtype)
        assertEquals("int64", engine.lastInputs?.get("attention_mask")?.dtype)
    }

    @Test
    fun e2T7CountTokensReturnsCountAndTruncationFlag() {
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(tokenCount = 7),
            onnxSession = ONNXSession(
                sessionId = "mini-embed",
                engine = MockONNXEngine(),
                metadata = ONNXModelMetadataValue(
                    inputs = emptyList(),
                    outputs = emptyList(),
                    accelerator = "cpu",
                    opset = 17,
                ),
                sessionPriority = SessionPriority.INTERACTIVE,
            ),
            config = EmbeddingSessionConfig(
                dims = 2,
                maxSequenceLength = 6,
                tokenizerType = "wordpiece",
                pooling = "mean",
                normalize = true,
            ),
        )

        val result = session.countTokens("hello")

        assertEquals(7, result.count)
        assertTrue(result.truncated)
    }

    @Test
    fun e2T5NormalizeFalseReturnsRawPooledVector() {
        val engine = MockONNXEngine()
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = ONNXSession(
                sessionId = "mini-embed",
                engine = engine,
                metadata = ONNXModelMetadataValue(
                    inputs = engine.inputMetadata,
                    outputs = engine.outputMetadata,
                    accelerator = "cpu",
                    opset = 17,
                ),
                sessionPriority = SessionPriority.INTERACTIVE,
            ),
            config = EmbeddingSessionConfig(
                dims = 2,
                maxSequenceLength = 6,
                tokenizerType = "wordpiece",
                pooling = "mean",
                normalize = false,
            ),
        )

        val result = session.embed("hello")

        val norm = sqrt(result.embedding[0] * result.embedding[0] + result.embedding[1] * result.embedding[1])
        assertNotEquals(1.0f, norm, 0.001f)
    }

    @Test
    fun e2T8InferenceErrorIsPropagated() {
        val engine = MockONNXEngine(
            errorToThrow = RuntimeException("GPU out of memory"),
        )
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = ONNXSession(
                sessionId = "mini-embed",
                engine = engine,
                metadata = ONNXModelMetadataValue(
                    inputs = engine.inputMetadata,
                    outputs = engine.outputMetadata,
                    accelerator = "cpu",
                    opset = 17,
                ),
                sessionPriority = SessionPriority.INTERACTIVE,
            ),
            config = makeConfig(),
        )

        val error = assertThrows(ONNXError.InferenceError::class.java) {
            session.embed("hello")
        }
        assertTrue(error.message!!.contains("GPU out of memory"))
    }

    @Test
    fun e2T9EvictedSessionThrowsModelEvicted() {
        val engine = MockONNXEngine()
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = ONNXSession(
                sessionId = "mini-embed",
                engine = engine,
                metadata = ONNXModelMetadataValue(
                    inputs = engine.inputMetadata,
                    outputs = engine.outputMetadata,
                    accelerator = "cpu",
                    opset = 17,
                ),
                sessionPriority = SessionPriority.INTERACTIVE,
            ),
            config = makeConfig(),
        )

        session.evict()

        assertThrows(ONNXError.ModelEvicted::class.java) {
            session.embed("hello")
        }
    }

    @Test
    fun e3T1EmbedBatchPadsShorterInputsToChunkMaxLength() {
        val engine = MockONNXEngine()
        val session = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = engine,
        )

        session.embedBatch(listOf("Hello", "Hello world"))

        assertEquals(listOf(2, 4), engine.lastInputs?.get("input_ids")?.shape)
        assertEquals(
            listOf(101.0, 200.0, 102.0, 0.0, 101.0, 200.0, 201.0, 102.0),
            engine.lastInputs?.get("input_ids")?.data,
        )
        assertEquals(
            listOf(1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            engine.lastInputs?.get("attention_mask")?.data,
        )
    }

    @Test
    fun e3T2EmbedBatchReturnsNormalizedEmbeddingsWithConfiguredDims() {
        val session = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = MockONNXEngine(),
        )

        val results = session.embedBatch(listOf("Hello", "Hello world"))

        assertEquals(2, results.size)
        results.forEach { result ->
            assertEquals(2, result.embedding.size)
            val norm = sqrt(result.embedding[0] * result.embedding[0] + result.embedding[1] * result.embedding[1])
            assertEquals(1.0f, norm, 0.0001f)
        }
    }

    @Test
    fun e3T3EmbedBatchSingleItemMatchesSingleEmbed() {
        val singleSession = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = MockONNXEngine(),
        )
        val batchSession = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = MockONNXEngine(),
        )

        val single = singleSession.embed("Hello")
        val batch = batchSession.embedBatch(listOf("Hello")).single()

        assertEquals(single.tokenCount, batch.tokenCount)
        assertEquals(single.truncated, batch.truncated)
        assertEquals(single.modelId, batch.modelId)
        single.embedding.zip(batch.embedding).forEach { (expected, actual) ->
            assertEquals(expected, actual, 0.0001f)
        }
    }

    @Test
    fun e3T4EmbedBatchReturnsEmptyWithoutRunningInference() {
        val engine = MockONNXEngine()
        val session = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = engine,
        )

        val results = session.embedBatch(emptyList())

        assertTrue(results.isEmpty())
        assertEquals(0, engine.runCallCount)
    }

    @Test
    fun e3T5EmbedBatchMarksTruncatedInputsWhenAllowed() {
        val session = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = MockONNXEngine(),
            config = makeConfig(maxSequenceLength = 4),
        )

        val results = session.embedBatch(listOf("Hello", "overflow_text"), truncate = true)

        assertEquals(2, results.size)
        assertTrue(results[1].truncated)
    }

    @Test
    fun e3T6EmbedBatchFailsFastWhenTruncationIsDisabled() {
        val engine = MockONNXEngine()
        val session = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = engine,
            config = makeConfig(maxSequenceLength = 4),
        )

        assertThrows(DustCoreError.InvalidInput::class.java) {
            session.embedBatch(listOf("Hello", "overflow_text"), truncate = false)
        }
        assertEquals(0, engine.runCallCount)
    }

    @Test
    fun e3T7EmbedBatchChunksBatchesLargerThan64Items() {
        val engine = MockONNXEngine()
        val session = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = engine,
        )

        val results = session.embedBatch(List(65) { "Hello" })

        assertEquals(65, results.size)
        assertEquals(2, engine.runCallCount)
    }

    @Test
    fun e3T8EmbedBatchWrapsBatchInferenceErrors() {
        val engine = MockONNXEngine(
            outputGenerator = {
                throw IllegalStateException("synthetic batch failure")
            },
        )
        val session = makeSession(
            tokenizer = BatchMockTokenizer(),
            engine = engine,
        )

        val error = assertThrows(DustCoreError.InferenceFailed::class.java) {
            session.embedBatch(listOf("Hello", "Hello world"))
        }

        assertTrue(error.message!!.contains("Batch inference failed"))
    }

    @Test
    fun e4T1GGUFEmbedReturnsVectorFromEngine() {
        val session = EmbeddingSession(
            sessionId = "gguf-embed",
            tokenizer = MockTokenizer(),
            onnxSession = null,
            config = makeConfig(dims = 4, maxSequenceLength = 32),
            ggufEngine = MockGGUFEngine(dims = 4),
        )

        val result = session.embed("hello")

        assertEquals("gguf-embed", result.modelId)
        assertEquals(4, result.embedding.size)
        assertTrue(result.tokenCount > 0)
    }

    @Test
    fun e4T2GGUFEmbedDoesNotCallONNXSession() {
        val session = EmbeddingSession(
            sessionId = "gguf-embed",
            tokenizer = MockTokenizer(),
            onnxSession = null,
            config = makeConfig(dims = 4, maxSequenceLength = 32),
            ggufEngine = MockGGUFEngine(dims = 4),
        )

        val result = session.embed("hello")

        assertEquals(4, result.embedding.size)
    }

    @Test
    fun e4T3GGUFCountTokensUsesEngineNotTokenizer() {
        val session = EmbeddingSession(
            sessionId = "gguf-embed",
            tokenizer = MockTokenizer(tokenCount = 3),
            onnxSession = null,
            config = makeConfig(dims = 4, maxSequenceLength = 64),
            ggufEngine = MockGGUFEngine(dims = 4, tokenCounts = { 42 }),
        )

        val result = session.countTokens("anything")

        assertEquals(42, result.count)
    }

    @Test
    fun e4T4GGUFEmbedBatchFallsBackToSequential() {
        val engine = MockGGUFEngine(dims = 4)
        val session = EmbeddingSession(
            sessionId = "gguf-embed",
            tokenizer = MockTokenizer(),
            onnxSession = null,
            config = makeConfig(dims = 4, maxSequenceLength = 32),
            ggufEngine = engine,
        )

        val results = session.embedBatch(listOf("a", "b", "c"))

        assertEquals(3, results.size)
        assertEquals(3, engine.embedCallCount)
    }

    @Test
    fun e4T5GGUFTruncateFalseThrowsWhenExceedsMax() {
        val session = EmbeddingSession(
            sessionId = "gguf-embed",
            tokenizer = MockTokenizer(),
            onnxSession = null,
            config = makeConfig(dims = 4, maxSequenceLength = 4),
            ggufEngine = MockGGUFEngine(dims = 4, tokenCounts = { 10 }),
        )

        assertThrows(DustCoreError.InvalidInput::class.java) {
            session.embed("long", truncate = false)
        }
    }

    @Test
    fun e4T6GGUFTruncateTrueMarksTruncated() {
        val session = EmbeddingSession(
            sessionId = "gguf-embed",
            tokenizer = MockTokenizer(),
            onnxSession = null,
            config = makeConfig(dims = 4, maxSequenceLength = 4),
            ggufEngine = MockGGUFEngine(dims = 4, tokenCounts = { 10 }),
        )

        val result = session.embed("long", truncate = true)

        assertTrue(result.truncated)
    }

    @Test
    fun e4T7GGUFEngineErrorWrappedAsInferenceFailed() {
        val session = EmbeddingSession(
            sessionId = "gguf-embed",
            tokenizer = MockTokenizer(),
            onnxSession = null,
            config = makeConfig(dims = 4, maxSequenceLength = 32),
            ggufEngine = MockGGUFEngine(
                dims = 4,
                errorToThrow = RuntimeException("GPU OOM"),
            ),
        )

        val error = assertThrows(DustCoreError.InferenceFailed::class.java) {
            session.embed("hello")
        }

        assertTrue(error.message!!.contains("Embedding extraction failed"))
    }

    @Test
    fun e4T8GGUFEvictReleasesEngine() {
        val engine = MockGGUFEngine(dims = 4)
        val session = EmbeddingSession(
            sessionId = "gguf-embed",
            tokenizer = MockTokenizer(),
            onnxSession = null,
            config = makeConfig(dims = 4, maxSequenceLength = 32),
            ggufEngine = engine,
        )

        session.evict()

        assertTrue(engine.evicted)
        assertThrows(ONNXError.ModelEvicted::class.java) {
            session.embed("hello")
        }
    }

    @Test
    fun e5T1ImageEmbedReturnsNormalizedVector() {
        val engine = makeImageEngine(
            dims = 4,
            cannedOutput = imageOutput(
                shape = listOf(1, 4),
                values = listOf(3.0, 4.0, 0.0, 0.0),
            ),
        )
        val onnxSession = makeONNXSession(engine)
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = onnxSession,
            config = makeConfig(dims = 4),
        )

        val result = session.inferImageTensor(
            session = onnxSession,
            tensorName = "pixel_values",
            preprocessed = makeImageTensor(),
        )

        assertEquals(4, result.embedding.size)
        assertEquals(0.6f, result.embedding[0], 0.0001f)
        assertEquals(0.8f, result.embedding[1], 0.0001f)
        assertEquals(0, result.tokenCount)
        assertFalse(result.truncated)
        assertEquals("mini-embed", result.modelId)
        assertEquals(listOf(1, 3, 224, 224), engine.lastInputs?.get("pixel_values")?.shape)
    }

    @Test
    fun e5T2ImageEmbedPools3DOutputTensor() {
        val engine = makeImageEngine(
            dims = 4,
            cannedOutput = imageOutput(
                shape = listOf(1, 2, 4),
                values = listOf(
                    2.0, 0.0, 0.0, 0.0,
                    0.0, 2.0, 0.0, 0.0,
                ),
            ),
        )
        val onnxSession = makeONNXSession(engine)
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = onnxSession,
            config = makeConfig(dims = 4),
        )

        val result = session.inferImageTensor(
            session = onnxSession,
            tensorName = "pixel_values",
            preprocessed = makeImageTensor(),
        )

        val expected = sqrt(0.5f)
        assertEquals(expected, result.embedding[0], 0.0001f)
        assertEquals(expected, result.embedding[1], 0.0001f)
        assertEquals(0f, result.embedding[2], 0.0001f)
        assertEquals(0f, result.embedding[3], 0.0001f)
    }

    @Test
    fun e5T3ImageEmbedNormalizeFalseReturnsRawVector() {
        val engine = makeImageEngine(
            dims = 4,
            cannedOutput = imageOutput(
                shape = listOf(1, 4),
                values = listOf(3.0, 4.0, 0.0, 0.0),
            ),
        )
        val onnxSession = makeONNXSession(engine)
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = onnxSession,
            config = makeConfig(dims = 4, normalize = false),
        )

        val result = session.inferImageTensor(
            session = onnxSession,
            tensorName = "pixel_values",
            preprocessed = makeImageTensor(),
        )

        assertEquals(listOf(3f, 4f, 0f, 0f), result.embedding)
    }

    @Test
    fun e5T4ResolveImageInputFinds4DTensor() {
        val session = makeSession()

        val imageInput = session.resolveImageInput(
            listOf(
                ONNXTensorMetadata("input_ids", listOf(-1, -1), "int64"),
                ONNXTensorMetadata("pixel_values", listOf(1, 3, 128, 256), "float32"),
                ONNXTensorMetadata("attention_mask", listOf(-1, -1), "int64"),
            ),
        )

        assertEquals("pixel_values", imageInput.metadata.name)
        assertEquals(256, imageInput.width)
        assertEquals(128, imageInput.height)
    }

    @Test
    fun e5T5ResolveImageInputDefaultsTo224WhenDynamic() {
        val session = makeSession()

        val imageInput = session.resolveImageInput(
            listOf(
                ONNXTensorMetadata("pixel_values", listOf(-1, 3, -1, -1), "float32"),
            ),
        )

        assertEquals(224, imageInput.width)
        assertEquals(224, imageInput.height)
    }

    @Test
    fun e5T6ResolveImageInputThrowsWhenNoImageTensor() {
        val session = makeSession()

        val error = assertThrows(DustCoreError.InvalidInput::class.java) {
            session.resolveImageInput(
                listOf(
                    ONNXTensorMetadata("input_ids", listOf(-1, -1), "int64"),
                    ONNXTensorMetadata("attention_mask", listOf(-1, -1), "int64"),
                ),
            )
        }

        assertTrue(error.message!!.contains("Model does not expose an image input tensor"))
    }

    @Test
    fun e5T7ImageEmbedInferenceErrorPropagates() {
        val engine = makeImageEngine(
            dims = 4,
            errorToThrow = RuntimeException("synthetic image failure"),
        )
        val onnxSession = makeONNXSession(engine)
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = onnxSession,
            config = makeConfig(dims = 4),
        )

        val error = assertThrows(ONNXError.InferenceError::class.java) {
            session.inferImageTensor(
                session = onnxSession,
                tensorName = "pixel_values",
                preprocessed = makeImageTensor(),
            )
        }

        assertTrue(error.message!!.contains("synthetic image failure"))
    }

    @Test
    fun e5T8ImageEmbedDimensionMismatchThrows() {
        val engine = makeImageEngine(
            dims = 8,
            cannedOutput = imageOutput(
                shape = listOf(1, 8),
                values = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
            ),
        )
        val onnxSession = makeONNXSession(engine)
        val session = EmbeddingSession(
            sessionId = "mini-embed",
            tokenizer = MockTokenizer(),
            onnxSession = onnxSession,
            config = makeConfig(dims = 4),
        )

        val error = assertThrows(DustCoreError.InferenceFailed::class.java) {
            session.inferImageTensor(
                session = onnxSession,
                tensorName = "pixel_values",
                preprocessed = makeImageTensor(),
            )
        }

        assertTrue(error.message!!.contains("Expected embedding dimension 4, got 8"))
    }

    private fun makeConfig(
        dims: Int = 2,
        maxSequenceLength: Int = 6,
        normalize: Boolean = true,
    ): EmbeddingSessionConfig = EmbeddingSessionConfig(
        dims = dims,
        maxSequenceLength = maxSequenceLength,
        tokenizerType = "wordpiece",
        pooling = "mean",
        normalize = normalize,
    )

    private fun makeSession(
        tokenizer: EmbeddingTokenizer = MockTokenizer(),
        engine: MockONNXEngine = MockONNXEngine(),
        config: EmbeddingSessionConfig = makeConfig(),
    ): EmbeddingSession = EmbeddingSession(
        sessionId = "mini-embed",
        tokenizer = tokenizer,
        onnxSession = makeONNXSession(engine),
        config = config,
    )

    private fun makeONNXSession(
        engine: MockONNXEngine,
        sessionId: String = "mini-embed",
    ): ONNXSession = ONNXSession(
        sessionId = sessionId,
        engine = engine,
        metadata = ONNXModelMetadataValue(
            inputs = engine.inputMetadata,
            outputs = engine.outputMetadata,
            accelerator = "cpu",
            opset = 17,
        ),
        sessionPriority = SessionPriority.INTERACTIVE,
    )

    private fun makeImageEngine(
        dims: Int = 4,
        imageWidth: Int = 224,
        imageHeight: Int = 224,
        cannedOutput: Map<String, TensorData>? = null,
        outputGenerator: ((Map<String, TensorData>) -> Map<String, TensorData>)? = null,
        errorToThrow: Exception? = null,
    ): MockONNXEngine = MockONNXEngine(
        inputMetadata = listOf(
            ONNXTensorMetadata("pixel_values", listOf(1, 3, imageHeight, imageWidth), "float32"),
        ),
        outputMetadata = listOf(
            ONNXTensorMetadata("last_hidden_state", listOf(1, dims), "float32"),
        ),
        cannedOutput = cannedOutput,
        outputGenerator = outputGenerator,
        errorToThrow = errorToThrow,
    )

    private fun makeImageTensor(
        width: Int = 224,
        height: Int = 224,
    ): TensorData = TensorData(
        name = "pixel_values",
        dtype = "float32",
        shape = listOf(1, 3, height, width),
        data = List(3 * height * width) { 0.5 },
    )

    private fun imageOutput(
        shape: List<Int>,
        values: List<Double>,
        name: String = "last_hidden_state",
    ): Map<String, TensorData> = mapOf(
        name to TensorData(
            name = name,
            dtype = "float32",
            shape = shape,
            data = values,
        ),
    )
}

internal class MockTokenizer(
    private val tokenCount: Int = 3,
) : EmbeddingTokenizer {
    override val vocabSize: Int = 1_000

    override fun tokenize(text: String, maxLength: Int): TokenizerOutput {
        val inputIds = listOf(101, 200, 102, 0).take(maxLength)
        val attentionMask = listOf(1, 1, 1, 0).take(maxLength)
        val tokenTypeIds = listOf(0, 0, 0, 0).take(maxLength)
        return TokenizerOutput(inputIds, attentionMask, tokenTypeIds)
    }

    override fun countTokens(text: String): Int = tokenCount
}

internal class BatchMockTokenizer : EmbeddingTokenizer {
    override val vocabSize: Int = 1_000

    override fun tokenize(text: String, maxLength: Int): TokenizerOutput {
        val inputIds = fullInputIds(text).take(maxLength)
        return TokenizerOutput(
            inputIds = inputIds,
            attentionMask = List(inputIds.size) { 1 },
            tokenTypeIds = List(inputIds.size) { 0 },
        )
    }

    override fun countTokens(text: String): Int = fullInputIds(text).size

    private fun fullInputIds(text: String): List<Int> = when {
        text == "overflow_text" -> listOf(101, 200, 201, 202, 203, 204, 205, 102)
        text.contains(" ") -> listOf(101, 200, 201, 102)
        else -> listOf(101, 200, 102)
    }
}

internal class MockGGUFEngine(
    override val dims: Int = 4,
    var errorToThrow: Throwable? = null,
    private val tokenCounts: (String) -> Int = { it.split(" ").size + 2 },
    private val embeddings: (String) -> FloatArray = { text ->
        val hash = text.hashCode()
        FloatArray(dims) { index -> ((hash shr (index * 8)) and 0xFF).toFloat() / 255f }
    },
) : GGUFEmbeddingEngine {
    var embedCallCount = 0
        private set
    var closed = false
        private set
    var evicted = false
        private set

    override fun embed(text: String): FloatArray {
        embedCallCount += 1
        errorToThrow?.let { throw it }
        return embeddings(text)
    }

    override fun countTokens(text: String): Int = tokenCounts(text)

    override fun close() {
        closed = true
    }

    override fun evict() {
        evicted = true
    }
}
