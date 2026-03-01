package io.t6x.dust.embeddings

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.EmbeddingStatus
import io.t6x.dust.core.SessionPriority
import io.t6x.dust.onnx.MemoryPressureLevel
import io.t6x.dust.onnx.ONNXConfig
import io.t6x.dust.onnx.ONNXModelMetadataValue
import io.t6x.dust.onnx.ONNXSession
import io.t6x.dust.onnx.ONNXSessionManager
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class EmbeddingSessionManagerTest {
    @Test
    fun e2T8LoadAndUnloadUpdatesRefCount() = runTest {
        val manager = makeManager()
        val modelFile = createTempModelFile()

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
        val modelFile = createTempModelFile()

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

    @Test
    fun e6T1LoadGGUFModelCachesSession() {
        val manager = makeManager()
        val config = makeConfig(dims = 4)
        val engine1 = MockGGUFEngine()

        val first = manager.loadGGUFModel(
            modelId = "gguf-model",
            engine = engine1,
            config = config,
        )

        assertTrue(manager.hasCachedSession("gguf-model"))
        assertEquals(1, manager.sessionCount)
        assertSame(first, manager.session("gguf-model"))
        assertEquals(1, manager.refCount("gguf-model"))

        val engine2 = MockGGUFEngine()
        val second = manager.loadGGUFModel(
            modelId = "gguf-model",
            engine = engine2,
            config = config,
        )

        assertSame(first, second)
        assertEquals(2, manager.refCount("gguf-model"))
        assertTrue(engine2.closed)
        assertFalse(engine1.closed)
    }

    @Test
    fun e6T2ForceUnloadRemovesFromCache() = runTest {
        val manager = makeManager()
        val modelFile = createTempModelFile()

        manager.loadModel(
            modelPath = modelFile.path,
            modelId = "mini-embed",
            vocabPath = "unused",
            mergesPath = null,
            config = makeConfig(),
            onnxConfig = ONNXConfig(),
            priority = SessionPriority.INTERACTIVE,
        )

        manager.forceUnloadModel("mini-embed")

        assertFalse(manager.hasCachedSession("mini-embed"))
        assertEquals(0, manager.sessionCount)
        assertNull(manager.session("mini-embed"))
    }

    @Test
    fun e6T3ForceUnloadUnknownModelThrows() {
        val manager = makeManager()

        assertThrows(DustCoreError.ModelNotFound::class.java) {
            runTest {
                manager.forceUnloadModel("nonexistent")
            }
        }
    }

    @Test
    fun e6T4UnloadUnknownModelThrows() {
        val manager = makeManager()

        assertThrows(DustCoreError.ModelNotFound::class.java) {
            manager.unloadModel("nonexistent")
        }
    }

    @Test
    fun e6T5AllModelIdsReturnsSortedIds() {
        val manager = makeManager()

        listOf("charlie", "alpha", "bravo").forEach { id ->
            val modelFile = createTempModelFile()
            manager.loadModel(
                modelPath = modelFile.path,
                modelId = id,
                vocabPath = "unused",
                mergesPath = null,
                config = makeConfig(),
                onnxConfig = ONNXConfig(),
                priority = SessionPriority.INTERACTIVE,
            )
        }

        assertEquals(listOf("alpha", "bravo", "charlie"), manager.allModelIds())
    }

    @Test
    fun e6T6StatusIdleWhenEmptyReadyWhenLoaded() = runTest {
        val manager = makeManager()

        assertEquals(EmbeddingStatus.IDLE, manager.status())

        val modelFile = createTempModelFile()
        manager.loadModel(
            modelPath = modelFile.path,
            modelId = "mini-embed",
            vocabPath = "unused",
            mergesPath = null,
            config = makeConfig(),
            onnxConfig = ONNXConfig(),
            priority = SessionPriority.INTERACTIVE,
        )
        assertEquals(EmbeddingStatus.READY, manager.status())

        manager.forceUnloadModel("mini-embed")

        assertEquals(EmbeddingStatus.IDLE, manager.status())
    }

    @Test
    fun e6T7EmbeddingDimensionReturnsConfiguredDims() = runTest {
        val manager = makeManager()

        assertEquals(0, manager.embeddingDimension())

        val modelFile = createTempModelFile()
        manager.loadModel(
            modelPath = modelFile.path,
            modelId = "mini-embed",
            vocabPath = "unused",
            mergesPath = null,
            config = makeConfig(dims = 4),
            onnxConfig = ONNXConfig(),
            priority = SessionPriority.INTERACTIVE,
        )
        assertEquals(4, manager.embeddingDimension())

        manager.forceUnloadModel("mini-embed")

        assertEquals(0, manager.embeddingDimension())
    }

    @Test
    fun e6T8StandardPressureOnlyEvictsBackgroundSessions() = runTest {
        val manager = makeManager()
        val interactiveFile = createTempModelFile()
        val backgroundFile = createTempModelFile()

        manager.loadModel(
            modelPath = interactiveFile.path,
            modelId = "interactive-model",
            vocabPath = "unused",
            mergesPath = null,
            config = makeConfig(),
            onnxConfig = ONNXConfig(),
            priority = SessionPriority.INTERACTIVE,
        )
        manager.loadModel(
            modelPath = backgroundFile.path,
            modelId = "background-model",
            vocabPath = "unused",
            mergesPath = null,
            config = makeConfig(),
            onnxConfig = ONNXConfig(),
            priority = SessionPriority.BACKGROUND,
        )
        manager.unloadModel("interactive-model")
        manager.unloadModel("background-model")

        manager.evictUnderPressure(MemoryPressureLevel.STANDARD)

        assertTrue(manager.hasCachedSession("interactive-model"))
        assertFalse(manager.hasCachedSession("background-model"))
        assertEquals(1, manager.sessionCount)
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

    private fun makeConfig(dims: Int = 2): EmbeddingSessionConfig = EmbeddingSessionConfig(
        dims = dims,
        maxSequenceLength = 8,
        tokenizerType = "wordpiece",
        pooling = "mean",
        normalize = true,
    )

    private fun createTempModelFile(): File = File.createTempFile("mini-embed-", ".onnx").apply {
        writeText("fixture")
        deleteOnExit()
    }
}
