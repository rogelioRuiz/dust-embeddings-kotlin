package io.t6x.dust.embeddings

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.onnx.ImagePreprocessor
import io.t6x.dust.onnx.ONNXError
import io.t6x.dust.onnx.ONNXSession
import io.t6x.dust.onnx.ONNXTensorMetadata
import io.t6x.dust.onnx.TensorData
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

data class EmbeddingResult(
    val embedding: List<Float>,
    val tokenCount: Int,
    val truncated: Boolean,
    val modelId: String,
)

data class TokenCountResult(
    val count: Int,
    val truncated: Boolean,
)

class EmbeddingSession(
    val sessionId: String,
    private val tokenizer: EmbeddingTokenizer,
    private var onnxSession: ONNXSession?,
    val config: EmbeddingSessionConfig,
    private var ggufEngine: GGUFEmbeddingEngine? = null,
) {
    private val lock = ReentrantLock()
    private val ggufBackend = ggufEngine != null
    private var evicted = false

    fun embed(text: String, truncate: Boolean = true): EmbeddingResult {
        if (ggufBackend) {
            val engine = activeGGUFEngine()
            val rawTokenCount = engine.countTokens(text)
            val truncated = rawTokenCount > ggufMaxTokenCount
            if (truncated && !truncate) {
                throw DustCoreError.InvalidInput("Input exceeds maxSequenceLength of ${config.maxSequenceLength}")
            }

            val embedding = try {
                engine.embed(text).toList()
            } catch (error: Throwable) {
                throw DustCoreError.InferenceFailed("Embedding extraction failed: ${error.message ?: error}")
            }

            validate(embedding)

            return EmbeddingResult(
                embedding = embedding,
                tokenCount = rawTokenCount,
                truncated = truncated,
                modelId = sessionId,
            )
        }

        val rawTokenCount = tokenizer.countTokens(text)
        val truncated = rawTokenCount > maxTokenCount
        if (truncated && !truncate) {
            throw DustCoreError.InvalidInput("Input exceeds maxSequenceLength of ${config.maxSequenceLength}")
        }

        val tokenOutput = tokenizer.tokenize(text, config.maxSequenceLength)
        val outputTensor = runTextInference(tokenOutput)
        val embedding = extractEmbedding(
            outputTensor,
            normalizedMask(tokenOutput.attentionMask, sequenceLength(outputTensor)),
        ).toMutableList()

        if (config.normalize) {
            VectorMath.l2Normalize(embedding)
        }

        return EmbeddingResult(
            embedding = embedding,
            tokenCount = rawTokenCount,
            truncated = truncated,
            modelId = sessionId,
        )
    }

    fun embedBatch(texts: List<String>, truncate: Boolean = true): List<EmbeddingResult> =
        if (texts.isEmpty()) {
            emptyList()
        } else if (ggufBackend) {
            texts.map { embed(it, truncate) }
        } else {
            texts.chunked(BATCH_SIZE).flatMap { embedBatchChunk(it, truncate) }
        }

    fun embedImage(imageData: ByteArray): EmbeddingResult {
        val session = activeSession()
        val imageInput = resolveImageInput(session.metadata.inputs)
        val preprocessed = ImagePreprocessor.preprocess(
            imageData = imageData,
            targetWidth = imageInput.width,
            targetHeight = imageInput.height,
            resize = "crop_center",
            normalization = "imagenet",
            customMean = null,
            customStd = null,
        )
        return inferImageTensor(
            session = session,
            tensorName = imageInput.metadata.name,
            preprocessed = preprocessed,
        )
    }

    internal fun inferImageTensor(
        session: ONNXSession,
        tensorName: String,
        preprocessed: TensorData,
    ): EmbeddingResult {
        val imageTensor = TensorData(
            name = tensorName,
            dtype = preprocessed.dtype,
            shape = preprocessed.shape,
            data = preprocessed.data,
        )
        val outputs = session.runInference(
            inputs = mapOf(imageTensor.name to imageTensor),
            outputNames = listOf(config.outputName),
        )
        val outputTensor = outputs[config.outputName]
            ?: throw DustCoreError.InferenceFailed("Output tensor '${config.outputName}' was not returned")

        val embedding = extractEmbedding(
            outputTensor,
            List(sequenceLength(outputTensor)) { 1 },
        ).toMutableList()
        if (config.normalize) {
            VectorMath.l2Normalize(embedding)
        }

        return EmbeddingResult(
            embedding = embedding,
            tokenCount = 0,
            truncated = false,
            modelId = sessionId,
        )
    }

    fun countTokens(text: String): TokenCountResult {
        val count = if (ggufBackend) {
            activeGGUFEngine().countTokens(text)
        } else {
            tokenizer.countTokens(text)
        }
        val maxCount = if (ggufBackend) ggufMaxTokenCount else maxTokenCount
        return TokenCountResult(count, count > maxCount)
    }

    fun tokenize(text: String, maxLength: Int? = null): TokenizerOutput =
        tokenizer.tokenize(text, maxLength ?: config.maxSequenceLength)

    suspend fun close() {
        if (ggufBackend) {
            val engine = lock.withLock {
                val current = ggufEngine
                ggufEngine = null
                onnxSession = null
                evicted = false
                current
            }
            engine?.close()
            return
        }

        val session = detach(evicted = false)
        session?.close()
    }

    fun evict() {
        if (ggufBackend) {
            val engine = lock.withLock {
                val current = ggufEngine
                ggufEngine = null
                onnxSession = null
                evicted = true
                current
            }
            engine?.evict()
            return
        }

        val session = detach(evicted = true)
        session?.evict()
    }

    internal fun detach(evicted: Boolean): ONNXSession? = lock.withLock {
        val session = onnxSession
        onnxSession = null
        this.evicted = evicted
        session
    }

    private val maxTokenCount: Int
        get() = (config.maxSequenceLength - 2).coerceAtLeast(0)

    internal val usesGGUFBackend: Boolean
        get() = ggufBackend

    private val ggufMaxTokenCount: Int
        get() = config.maxSequenceLength

    private fun activeGGUFEngine(): GGUFEmbeddingEngine = lock.withLock {
        ggufEngine ?: throw if (evicted) {
            ONNXError.ModelEvicted
        } else {
            DustCoreError.SessionClosed
        }
    }

    private fun activeSession(): ONNXSession = lock.withLock {
        onnxSession ?: throw if (evicted) {
            ONNXError.ModelEvicted
        } else {
            DustCoreError.SessionClosed
        }
    }

    private fun runTextInference(tokenOutput: TokenizerOutput): TensorData {
        val session = activeSession()
        val inputNames = session.metadata.inputs.map { it.name }.toSet()
        val seqLen = tokenOutput.inputIds.size

        val inputs = linkedMapOf<String, TensorData>()
        inputs[config.inputNames.inputIds] = TensorData(
            name = config.inputNames.inputIds,
            dtype = "int64",
            shape = listOf(1, seqLen),
            data = tokenOutput.inputIds.map(Int::toDouble),
        )
        if (inputNames.contains(config.inputNames.attentionMask)) {
            inputs[config.inputNames.attentionMask] = TensorData(
                name = config.inputNames.attentionMask,
                dtype = "int64",
                shape = listOf(1, seqLen),
                data = tokenOutput.attentionMask.map(Int::toDouble),
            )
        }
        if (inputNames.contains(config.inputNames.tokenTypeIds)) {
            inputs[config.inputNames.tokenTypeIds] = TensorData(
                name = config.inputNames.tokenTypeIds,
                dtype = "int64",
                shape = listOf(1, seqLen),
                data = tokenOutput.tokenTypeIds.map(Int::toDouble),
            )
        }

        val outputs = session.runInference(inputs, outputNames = listOf(config.outputName))
        return outputs[config.outputName]
            ?: throw DustCoreError.InferenceFailed("Output tensor '${config.outputName}' was not returned")
    }

    private fun embedBatchChunk(texts: List<String>, truncate: Boolean): List<EmbeddingResult> {
        val rawTokenCounts = texts.map(tokenizer::countTokens)
        if (!truncate && rawTokenCounts.any { it > maxTokenCount }) {
            throw DustCoreError.InvalidInput("Input exceeds maxSequenceLength of ${config.maxSequenceLength}")
        }

        val tokenOutputs = texts.map { tokenizer.tokenize(it, config.maxSequenceLength) }
        val maxSeqLen = tokenOutputs.maxOfOrNull { it.inputIds.size } ?: 0
        val paddedOutputs = tokenOutputs.map { padTokenizerOutput(it, maxSeqLen) }
        val outputTensor = runBatchTextInference(paddedOutputs, maxSeqLen)
        val embeddings = extractBatchEmbeddings(outputTensor, paddedOutputs, texts.size)

        return texts.indices.map { index ->
            val embedding = embeddings[index].toMutableList()
            if (config.normalize) {
                VectorMath.l2Normalize(embedding)
            }

            EmbeddingResult(
                embedding = embedding,
                tokenCount = rawTokenCounts[index],
                truncated = rawTokenCounts[index] > maxTokenCount,
                modelId = sessionId,
            )
        }
    }

    private fun padTokenizerOutput(tokenOutput: TokenizerOutput, targetLength: Int): TokenizerOutput {
        if (tokenOutput.inputIds.size >= targetLength) {
            return tokenOutput
        }

        val padding = List(targetLength - tokenOutput.inputIds.size) { 0 }
        return TokenizerOutput(
            inputIds = tokenOutput.inputIds + padding,
            attentionMask = tokenOutput.attentionMask + padding,
            tokenTypeIds = tokenOutput.tokenTypeIds + padding,
        )
    }

    private fun runBatchTextInference(
        tokenOutputs: List<TokenizerOutput>,
        seqLen: Int,
    ): TensorData {
        val session = activeSession()
        val inputNames = session.metadata.inputs.map { it.name }.toSet()
        val batchSize = tokenOutputs.size

        val inputs = linkedMapOf<String, TensorData>()
        inputs[config.inputNames.inputIds] = TensorData(
            name = config.inputNames.inputIds,
            dtype = "int64",
            shape = listOf(batchSize, seqLen),
            data = tokenOutputs.flatMap { it.inputIds }.map(Int::toDouble),
        )
        if (inputNames.contains(config.inputNames.attentionMask)) {
            inputs[config.inputNames.attentionMask] = TensorData(
                name = config.inputNames.attentionMask,
                dtype = "int64",
                shape = listOf(batchSize, seqLen),
                data = tokenOutputs.flatMap { it.attentionMask }.map(Int::toDouble),
            )
        }
        if (inputNames.contains(config.inputNames.tokenTypeIds)) {
            inputs[config.inputNames.tokenTypeIds] = TensorData(
                name = config.inputNames.tokenTypeIds,
                dtype = "int64",
                shape = listOf(batchSize, seqLen),
                data = tokenOutputs.flatMap { it.tokenTypeIds }.map(Int::toDouble),
            )
        }

        val outputs = try {
            session.runInference(inputs, outputNames = listOf(config.outputName))
        } catch (error: Throwable) {
            throw DustCoreError.InferenceFailed("Batch inference failed: ${error.message ?: error.toString()}")
        }

        return outputs[config.outputName]
            ?: throw DustCoreError.InferenceFailed("Output tensor '${config.outputName}' was not returned")
    }

    private fun extractBatchEmbeddings(
        outputTensor: TensorData,
        tokenOutputs: List<TokenizerOutput>,
        batchSize: Int,
    ): List<List<Float>> {
        if (outputTensor.shape.size == 2 && outputTensor.shape.firstOrNull() == batchSize) {
            val hiddenDim = outputTensor.shape[1]
            val values = outputTensor.data.map(Double::toFloat)
            return List(batchSize) { index ->
                val start = index * hiddenDim
                val embedding = values.subList(start, start + hiddenDim).toList()
                validate(embedding)
                embedding
            }
        }

        if (outputTensor.shape.size == 3 && outputTensor.shape[0] == batchSize) {
            val seqLen = outputTensor.shape[1]
            val hiddenDim = outputTensor.shape[2]
            val values = outputTensor.data.map(Double::toFloat)
            return List(batchSize) { index ->
                val start = index * seqLen * hiddenDim
                val end = start + (seqLen * hiddenDim)
                val pooled = Pooling.apply(
                    strategy = config.pooling,
                    hiddenStates = values.subList(start, end),
                    attentionMask = normalizedMask(tokenOutputs[index].attentionMask, seqLen),
                    seqLen = seqLen,
                    hiddenDim = hiddenDim,
                )
                validate(pooled)
                pooled
            }
        }

        throw DustCoreError.InferenceFailed("Unsupported embedding tensor shape: ${outputTensor.shape}")
    }

    private fun extractEmbedding(
        outputTensor: TensorData,
        attentionMask: List<Int>,
    ): List<Float> {
        if (outputTensor.shape.size == 2 && outputTensor.shape.firstOrNull() == 1) {
            val embedding = outputTensor.data.map(Double::toFloat)
            validate(embedding)
            return embedding
        }

        if (outputTensor.shape.size == 3 && outputTensor.shape[0] == 1) {
            val seqLen = outputTensor.shape[1]
            val hiddenDim = outputTensor.shape[2]
            val hiddenStates = outputTensor.data.map(Double::toFloat)
            val pooled = Pooling.apply(
                strategy = config.pooling,
                hiddenStates = hiddenStates,
                attentionMask = normalizedMask(attentionMask, seqLen),
                seqLen = seqLen,
                hiddenDim = hiddenDim,
            )
            validate(pooled)
            return pooled
        }

        throw DustCoreError.InferenceFailed("Unsupported embedding tensor shape: ${outputTensor.shape}")
    }

    private fun validate(embedding: List<Float>) {
        if (embedding.size != config.dims) {
            throw DustCoreError.InferenceFailed(
                "Expected embedding dimension ${config.dims}, got ${embedding.size}",
            )
        }
    }

    private fun sequenceLength(outputTensor: TensorData): Int =
        if (outputTensor.shape.size == 3) {
            outputTensor.shape[1].coerceAtLeast(1)
        } else {
            1
        }

    private fun normalizedMask(mask: List<Int>, count: Int): List<Int> {
        if (count <= 0) {
            return emptyList()
        }
        if (mask.size == count) {
            return mask
        }
        if (mask.size > count) {
            return mask.take(count)
        }
        return mask + List(count - mask.size) { 0 }
    }

    internal fun resolveImageInput(metadata: List<ONNXTensorMetadata>): ImageInput {
        val imageInput = metadata.firstOrNull { it.shape.size == 4 }
            ?: throw DustCoreError.InvalidInput("Model does not expose an image input tensor")
        val height = imageInput.shape[2].takeIf { it > 0 } ?: 224
        val width = imageInput.shape[3].takeIf { it > 0 } ?: 224
        return ImageInput(imageInput, width, height)
    }
}

private const val BATCH_SIZE = 64

internal data class ImageInput(
    val metadata: ONNXTensorMetadata,
    val width: Int,
    val height: Int,
)
