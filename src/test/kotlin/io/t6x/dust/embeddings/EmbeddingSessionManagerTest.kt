package io.t6x.dust.embeddings

import io.t6x.dust.core.SessionPriority
import io.t6x.dust.onnx.MemoryPressureLevel
import io.t6x.dust.onnx.ONNXConfig
import io.t6x.dust.onnx.ONNXModelMetadataValue
import io.t6x.dust.onnx.ONNXSession
import io.t6x.dust.onnx.ONNXSessionManager
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Test
import java.io.File

class EmbeddingSessionManagerTest {
    @Test
    fun e2T8LoadAndUnloadUpdatesRefCount() = runTest {
        val manager = makeManager()
        val modelFile = File.createTempFile("mini-embed-", ".onnx").apply {
            writeText("fixture")
            deleteOnExit()
        }

        val first = manager.loadModel(
            modelPath = modelFile.path,
            modelId = "mini-embed",
            vocabPath = "unused",
            mergesPath = null,
            config = makeConfig(),
            onnxConfig = ONNXConfig(),
            priority = SessionPriority.INTERACTIVE,
        )
        val second = manager.loadModel(
            modelPath = modelFile.path,
            modelId = "mini-embed",
            vocabPath = "unused",
            mergesPath = null,
            config = makeConfig(),
            onnxConfig = ONNXConfig(),
            priority = SessionPriority.INTERACTIVE,
        )

        assertSame(first, second)
        assertEquals(2, manager.refCount("mini-embed"))

        manager.unloadModel("mini-embed")

        assertEquals(1, manager.refCount("mini-embed"))
    }

    @Test
    fun e2T9CriticalPressureEvictsUnreferencedSessions() = runTest {
        val manager = makeManager()
        val modelFile = File.createTempFile("mini-embed-", ".onnx").apply {
            writeText("fixture")
            deleteOnExit()
        }

        manager.loadModel(
            modelPath = modelFile.path,
            modelId = "mini-embed",
            vocabPath = "unused",
            mergesPath = null,
            config = makeConfig(),
            onnxConfig = ONNXConfig(),
            priority = SessionPriority.BACKGROUND,
        )
        manager.unloadModel("mini-embed")

        manager.evictUnderPressure(MemoryPressureLevel.CRITICAL)

        assertFalse(manager.hasCachedSession("mini-embed"))
        assertEquals(0, manager.sessionCount)
    }

    private fun makeManager(): EmbeddingSessionManager {
        val onnxManager = ONNXSessionManager(
            sessionFactory = { _, modelId, _, priority ->
                ONNXSession(
                    sessionId = modelId,
                    metadata = ONNXModelMetadataValue(
                        inputs = emptyList(),
                        outputs = emptyList(),
                        accelerator = "cpu",
                        opset = 17,
                    ),
                    priority = priority,
                )
            },
        )

        return EmbeddingSessionManager(
            onnxSessionManager = onnxManager,
            sessionFactory = { onnxSession, modelId, config, _, _, _ ->
                EmbeddingSession(
                    sessionId = modelId,
                    tokenizer = MockTokenizer(),
                    onnxSession = onnxSession,
                    config = config,
                )
            },
        )
    }

    private fun makeConfig(): EmbeddingSessionConfig = EmbeddingSessionConfig(
        dims = 2,
        maxSequenceLength = 8,
        tokenizerType = "wordpiece",
        pooling = "mean",
        normalize = true,
    )
}
